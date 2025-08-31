"""
CBDB Multi-Agent RAG System

This module implements a sophisticated multi-agent RAG system specifically designed
for the CBDB (China Biographical Database) with 80,000+ Person nodes and hundreds
of relationship types with rich properties.

The system includes:
- Specialized retrievers for different types of CBDB queries
- Text2Cypher functionality for complex graph queries
- Retriever router for selecting optimal retrieval strategies
- Answer critic for ensuring response quality
- Context enhancement using LLM processing
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
import streamlit as st
from neo4j import GraphDatabase


@dataclass
class CBDBQuery:
    """Represents a CBDB query with context"""
    original_question: str
    processed_question: str
    query_type: str
    entities: List[str]
    time_period: Optional[str] = None
    relationship_types: Optional[List[str]] = None


class CBDBText2Cypher:
    """Advanced Text2Cypher specifically designed for CBDB schema"""
    
    def __init__(self, neo4j_driver, openai_client):
        self.driver = neo4j_driver
        self.client = openai_client
        
        # CBDB-specific schema information
        self.schema_context = """
        CBDB Schema Information:
        - Nodes: Person (80,000+ nodes)
        - Person properties: 
          * person_id: unique identifier
          * name: English name (e.g., "Zhao Shitao")
          * name_chn: Chinese name (e.g., "趙師淘")
          * surname: English surname (e.g., "Zhao")
          * surname_chn: Chinese surname (e.g., "趙")
          * mingzi: English given name (e.g., "Shitao")
          * mingzi_chn: Chinese given name (e.g., "師淘")
          * birth_year: birth year (integer)
          * death_year: death year (integer)
          * age_at_death: age at death (integer)
          * index_year: reference year (integer)
          * gender: "male" or "female"
          * notes: array of biographical notes with sources
        
        - Relationships: Complex relationship types with rich metadata
        - Relationship properties:
          * association_code: numeric code
          * association_type_parent_id: parent category ID
          * association_type_desc_chn: Chinese description
          * association_desc: English description
          * association_desc_chn: Chinese description
          * association_type_code: specific type code
          * association_type_parent_chn: Chinese parent category
          * association_type_desc: English type description
          * association_type_parent: English parent category
        
        Common relationship patterns:
        - (:Person)-[:Coalition_member_of]->(:Person)
        - (:Person)-[:Father_of]->(:Person)
        - (:Person)-[:Student_of]->(:Person)
        - (:Person)-[:Contemporary_of]->(:Person)
        
        Example queries should use:
        - toLower(p.name) CONTAINS toLower($name) for English names
        - toLower(p.name_chn) CONTAINS toLower($name) for Chinese names
        - p.birth_year, p.death_year for temporal queries
        - r.association_type_desc for relationship type descriptions
        """
    
    def generate_cypher(self, question: str, context: Optional[str] = None) -> str:
        """Generate Cypher query for CBDB-specific questions"""
        
        system_prompt = f"""
        You are an expert in generating Cypher queries for the CBDB (China Biographical Database).
        
        {self.schema_context}
        
        Guidelines for CBDB queries:
        1. Always use case-insensitive matching for names: 
           - toLower(p.name) CONTAINS toLower($name) for English names
           - toLower(p.name_chn) CONTAINS toLower($name) for Chinese names
        2. Consider both English and Chinese names, surnames, and given names
        3. Use temporal constraints with proper null checks:
           - WHERE p.birth_year IS NOT NULL AND p.birth_year >= $year
           - Use coalesce(p.birth_year, p.index_year) for ordering
        4. For dynasty queries, search in notes array: 
           - WHERE any(note in p.notes WHERE toLower(note) CONTAINS toLower($dynasty))
        5. Return rich information including person_id, both name formats, and biographical details
        6. For relationships, include association metadata:
           - r.association_desc, r.association_type_desc, r.association_type_parent
        7. Use OPTIONAL MATCH for relationships that may not exist
        8. Limit results appropriately (usually 10-20 for person queries)
        
        Common query patterns:
        - Person lookup: MATCH (p:Person) WHERE toLower(p.name) CONTAINS toLower($name) OR toLower(p.name_chn) CONTAINS toLower($name)
        - Relationship queries: MATCH (p1:Person)-[r]-(p2:Person) RETURN r.association_desc, r.association_type_desc
        - Temporal queries: WHERE p.birth_year IS NOT NULL AND p.birth_year >= $start_year
        - Dynasty queries: WHERE any(note in p.notes WHERE toLower(note) CONTAINS toLower($dynasty))
        
        Return only the Cypher query, no explanations.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a Cypher query for: {question}"}
        ]
        
        if context:
            messages.insert(-1, {"role": "user", "content": f"Additional context: {context}"})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            cypher_query = response.choices[0].message.content.strip()
            # Clean up the query (remove markdown formatting if present)
            cypher_query = re.sub(r'```cypher\s*\n?|```\s*\n?', '', cypher_query)
            return cypher_query.strip()
            
        except Exception as e:
            st.error(f"Error generating Cypher query: {str(e)}")
            return ""


class CBDBRetrievers:
    """Collection of specialized retrievers for CBDB data"""
    
    def __init__(self, neo4j_driver, text2cypher: CBDBText2Cypher):
        self.driver = neo4j_driver
        self.text2cypher = text2cypher
    
    def person_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Get detailed information about a person by name"""
        query = """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS toLower($name) 
           OR toLower(p.name_chn) CONTAINS toLower($name)
           OR toLower(p.surname) CONTAINS toLower($name)
           OR toLower(p.mingzi) CONTAINS toLower($name)
        RETURN p {
            .person_id, .name, .name_chn, .surname, .surname_chn,
            .mingzi, .mingzi_chn, .birth_year, .death_year, 
            .age_at_death, .index_year, .gender, .notes
        } AS person
        ORDER BY p.birth_year
        LIMIT 10
        """
        try:
            records, _, _ = self.driver.execute_query(query, name=name)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in person_by_name: {str(e)}")
            return []
    
    def person_relationships(self, name: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relationships for a specific person"""
        base_query = """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS toLower($name)
           OR toLower(p.name_chn) CONTAINS toLower($name)
        """
        
        if relationship_type:
            query = base_query + f"""
            MATCH (p)-[r:{relationship_type}]-(related:Person)
            RETURN p {{.person_id, .name, .name_chn}} AS person, 
                   type(r) AS relationship_type,
                   r {{.association_desc, .association_desc_chn, .association_type_desc, 
                       .association_type_parent, .association_code}} AS relationship_details,
                   related {{.person_id, .name, .name_chn, .birth_year, .death_year}} AS related_person
            ORDER BY related.birth_year
            LIMIT 20
            """
        else:
            query = base_query + """
            MATCH (p)-[r]-(related:Person)
            RETURN p {.person_id, .name, .name_chn} AS person, 
                   type(r) AS relationship_type,
                   r {.association_desc, .association_desc_chn, .association_type_desc, 
                       .association_type_parent, .association_code} AS relationship_details,
                   related {.person_id, .name, .name_chn, .birth_year, .death_year} AS related_person
            ORDER BY related.birth_year
            LIMIT 20
            """
        
        try:
            records, _, _ = self.driver.execute_query(query, name=name)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in person_relationships: {str(e)}")
            return []
    
    def people_by_dynasty(self, dynasty: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Get people from a specific dynasty (searches in notes for dynasty references)"""
        query = """
        MATCH (p:Person)
        WHERE any(note in p.notes WHERE toLower(note) CONTAINS toLower($dynasty))
        RETURN p {
            .person_id, .name, .name_chn, .surname, .surname_chn,
            .birth_year, .death_year, .notes
        } AS person
        ORDER BY p.birth_year
        LIMIT $limit
        """
        try:
            records, _, _ = self.driver.execute_query(query, dynasty=dynasty, limit=limit)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in people_by_dynasty: {str(e)}")
            return []
    
    def people_by_time_period(self, start_year: int, end_year: int, limit: int = 15) -> List[Dict[str, Any]]:
        """Get people who lived during a specific time period"""
        query = """
        MATCH (p:Person)
        WHERE (p.birth_year IS NOT NULL AND p.birth_year >= $start_year AND p.birth_year <= $end_year)
           OR (p.death_year IS NOT NULL AND p.death_year >= $start_year AND p.death_year <= $end_year)
           OR (p.birth_year IS NOT NULL AND p.death_year IS NOT NULL AND 
               p.birth_year <= $start_year AND p.death_year >= $end_year)
           OR (p.index_year IS NOT NULL AND p.index_year >= $start_year AND p.index_year <= $end_year)
        RETURN p {
            .person_id, .name, .name_chn, .surname, .surname_chn,
            .birth_year, .death_year, .age_at_death, .index_year, .notes
        } AS person
        ORDER BY coalesce(p.birth_year, p.index_year)
        LIMIT $limit
        """
        try:
            records, _, _ = self.driver.execute_query(query, start_year=start_year, end_year=end_year, limit=limit)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in people_by_time_period: {str(e)}")
            return []
    
    def text2cypher_query(self, question: str) -> List[Dict[str, Any]]:
        """Execute a text2cypher generated query"""
        try:
            cypher_query = self.text2cypher.generate_cypher(question)
            if not cypher_query:
                return []
            
            st.info(f"Generated Cypher: {cypher_query}")
            records, _, _ = self.driver.execute_query(cypher_query)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in text2cypher_query: {str(e)}")
            return []
    
    def people_by_relationship_type(self, relationship_desc: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Find people connected by a specific relationship type"""
        query = """
        MATCH (p1:Person)-[r]-(p2:Person)
        WHERE toLower(r.association_desc) CONTAINS toLower($relationship_desc)
           OR toLower(r.association_type_desc) CONTAINS toLower($relationship_desc)
           OR toLower(r.association_type_parent) CONTAINS toLower($relationship_desc)
        RETURN p1 {.person_id, .name, .name_chn, .birth_year, .death_year} AS person1,
               r {.association_desc, .association_desc_chn, .association_type_desc, 
                   .association_type_parent, .association_code} AS relationship_details,
               p2 {.person_id, .name, .name_chn, .birth_year, .death_year} AS person2,
               type(r) AS relationship_type
        ORDER BY p1.birth_year
        LIMIT $limit
        """
        try:
            records, _, _ = self.driver.execute_query(query, relationship_desc=relationship_desc, limit=limit)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in people_by_relationship_type: {str(e)}")
            return []

    def answer_from_context(self, answer: str) -> str:
        """Extract answer that's already provided in the context"""
        return answer


class CBDBRetrieverRouter:
    """Router to select the best retriever for CBDB queries"""
    
    def __init__(self, retrievers: CBDBRetrievers, openai_client):
        self.retrievers = retrievers
        self.client = openai_client
        
        # Define available tools for OpenAI function calling
        self.tools = {
            "person_by_name": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "person_by_name",
                        "description": "Get detailed biographical information about a person by their name (English or Chinese)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The person's name (English or Chinese)",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                },
                "function": self.retrievers.person_by_name
            },
            "person_relationships": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "person_relationships",
                        "description": "Get relationships for a specific person, optionally filtered by relationship type",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The person's name",
                                },
                                "relationship_type": {
                                    "type": "string",
                                    "description": "Optional: specific relationship type (e.g., 'FATHER_OF', 'SERVED_UNDER')",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                },
                "function": self.retrievers.person_relationships
            },
            "people_by_dynasty": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "people_by_dynasty",
                        "description": "Get people from a specific Chinese dynasty",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "dynasty": {
                                    "type": "string",
                                    "description": "The dynasty name (e.g., 'Tang', 'Song', 'Ming')",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default 15)",
                                    "default": 15
                                }
                            },
                            "required": ["dynasty"],
                        },
                    },
                },
                "function": self.retrievers.people_by_dynasty
            },
            "people_by_time_period": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "people_by_time_period",
                        "description": "Get people who lived during a specific time period",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "start_year": {
                                    "type": "integer",
                                    "description": "Start year of the time period",
                                },
                                "end_year": {
                                    "type": "integer",
                                    "description": "End year of the time period",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default 15)",
                                    "default": 15
                                }
                            },
                            "required": ["start_year", "end_year"],
                        },
                    },
                },
                "function": self.retrievers.people_by_time_period
            },
            "text2cypher_query": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "text2cypher_query",
                        "description": "Use for complex queries that don't fit other tools. Generates and executes Cypher queries for CBDB data",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The complex question to answer using the CBDB database",
                                }
                            },
                            "required": ["question"],
                        },
                    },
                },
                "function": self.retrievers.text2cypher_query
            },
            "people_by_relationship_type": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "people_by_relationship_type",
                        "description": "Find people connected by a specific relationship type (e.g., 'Coalition member', 'political', 'family')",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "relationship_desc": {
                                    "type": "string",
                                    "description": "The relationship type or description to search for",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default 15)",
                                    "default": 15
                                }
                            },
                            "required": ["relationship_desc"],
                        },
                    },
                },
                "function": self.retrievers.people_by_relationship_type
            },
            "answer_from_context": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "answer_from_context",
                        "description": "Use when the answer is already provided in the conversation context",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "The answer extracted from context",
                                }
                            },
                            "required": ["answer"],
                        },
                    },
                },
                "function": self.retrievers.answer_from_context
            }
        }
    
    def route_question(self, question: str, context: List[Dict[str, str]] = None) -> List[Any]:
        """Route question to appropriate retriever"""
        
        system_prompt = """
        You are an expert router for the CBDB (China Biographical Database) system.
        Your job is to choose the right tool to answer questions about Chinese historical figures.
        
        Guidelines:
        - Use person_by_name for basic biographical information about specific people
        - Use person_relationships for questions about specific person's connections
        - Use people_by_dynasty for questions about specific dynasties (search in notes)
        - Use people_by_time_period for questions about specific time periods or years
        - Use people_by_relationship_type for questions about specific relationship types or categories
        - Use text2cypher_query for complex queries that require custom database searches
        - Use answer_from_context when the answer is already in the conversation
        
        Choose the most specific tool available for the question.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if context:
            messages.extend(context)
        
        messages.append({
            "role": "user", 
            "content": f"Route this CBDB question to the appropriate tool: {question}"
        })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=[tool["description"] for tool in self.tools.values()],
                tool_choice="auto",
                temperature=0.1
            )
            
            if response.choices[0].message.tool_calls:
                return self._handle_tool_calls(response.choices[0].message.tool_calls)
            else:
                return [{"error": "No appropriate tool selected"}]
                
        except Exception as e:
            st.error(f"Error in routing: {str(e)}")
            return [{"error": str(e)}]
    
    def _handle_tool_calls(self, tool_calls) -> List[Any]:
        """Execute the selected tools"""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name in self.tools:
                try:
                    function_to_call = self.tools[function_name]["function"]
                    result = function_to_call(**function_args)
                    results.append(result)
                except Exception as e:
                    results.append({"error": f"Error calling {function_name}: {str(e)}"})
            else:
                results.append({"error": f"Unknown function: {function_name}"})
        
        return results


class CBDBAnswerCritic:
    """Critic to evaluate and improve CBDB responses"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def critique_answer(self, original_question: str, context: List[Dict[str, str]]) -> List[str]:
        """Critique the answer and suggest follow-up questions if needed"""
        
        system_prompt = """
        You are an expert critic for CBDB (China Biographical Database) responses.
        Your job is to evaluate if the original question has been fully answered.
        
        For CBDB data, complete answers should include:
        - Biographical details (birth/death years, dynasty, offices held)
        - Relationship information when relevant
        - Historical context when appropriate
        - Chinese names when available
        
        If the answer is incomplete, provide specific follow-up questions that would
        complete the response. Questions should be atomic, specific, and answerable
        from the CBDB database.
        
        If the answer is complete, return an empty list.
        
        Return response in JSON format:
        {
            "questions": ["question1", "question2"]
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if context:
            messages.extend(context)
        
        messages.append({
            "role": "user",
            "content": f"Original question: {original_question}\nEvaluate if this has been fully answered."
        })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("questions", [])
            
        except Exception as e:
            st.error(f"Error in answer critique: {str(e)}")
            return []


class CBDBAgenticRAG:
    """Main orchestrator for the CBDB multi-agent RAG system"""
    
    def __init__(self, neo4j_driver, openai_client):
        self.driver = neo4j_driver
        self.client = openai_client
        
        # Initialize components
        self.text2cypher = CBDBText2Cypher(neo4j_driver, openai_client)
        self.retrievers = CBDBRetrievers(neo4j_driver, self.text2cypher)
        self.router = CBDBRetrieverRouter(self.retrievers, openai_client)
        self.critic = CBDBAnswerCritic(openai_client)
    
    def process_query(self, question: str, max_iterations: int = 2) -> str:
        """Process a CBDB query through the multi-agent system"""
        
        context = []
        iteration = 0
        
        while iteration < max_iterations:
            # Route question and get results
            results = self.router.route_question(question, context)
            
            # Add results to context
            context.append({
                "role": "assistant",
                "content": f"Retrieved data for '{question}': {json.dumps(results, ensure_ascii=False, indent=2)}"
            })
            
            # Critique the answer
            follow_up_questions = self.critic.critique_answer(question, context)
            
            if not follow_up_questions:
                break
            
            # Process follow-up questions
            for follow_up in follow_up_questions:
                follow_up_results = self.router.route_question(follow_up, context)
                context.append({
                    "role": "assistant",
                    "content": f"Additional data for '{follow_up}': {json.dumps(follow_up_results, ensure_ascii=False, indent=2)}"
                })
            
            iteration += 1
        
        # Generate final response
        return self._generate_final_response(question, context)
    
    def _generate_final_response(self, question: str, context: List[Dict[str, str]]) -> str:
        """Generate the final enhanced response"""
        
        system_prompt = """
        You are a knowledgeable assistant specializing in Chinese historical biographical data.
        
        Your task is to provide a comprehensive, well-structured answer based on the retrieved
        CBDB data. Your response should:
        
        1. Be historically accurate and based only on the provided data
        2. Include relevant biographical details (names, dates, dynasties, offices)
        3. Explain relationships and historical context
        4. Use both English and Chinese names when available
        5. Be engaging and informative
        6. Clearly state if information is missing or unavailable
        
        Format your response in a clear, readable manner with proper structure.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        messages.extend(context)
        
        messages.append({
            "role": "user",
            "content": f"Based on all the retrieved information, provide a comprehensive answer to: {question}"
        })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
