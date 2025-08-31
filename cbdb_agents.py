"""
CBDB Multi-Agent RAG System

This module implements a simplified multi-agent RAG system for the CBDB 
(China Biographical Database) with non-kinship relationships only.

The system includes:
- Specialized retrievers for biographical queries
- Text2Cypher functionality for complex non-kinship relationship queries
- Retriever router for selecting optimal retrieval strategies
- Answer synthesis for ensuring response quality

Note: Data contains Song dynasty figures only. System excludes kinship relationships.
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
        - Nodes: Person (historical figures)
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
        
        - Relationships: NON-KINSHIP relationships only with rich metadata
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
        
        Common NON-KINSHIP relationship patterns:
        - (:Person)-[:writing_in_his_stead_done_by]->(:Person)
        - (:Person)-[:Student_of]->(:Person)
        - (:Person)-[:Contemporary_of]->(:Person)
        - (:Person)-[:Served_under]->(:Person)
        - (:Person)-[:Political_ally_of]->(:Person)
        - (:Person)-[:Literary_associate_of]->(:Person)
        
        IMPORTANT: Exclude all kinship relationships (Father_of, Mother_of, Son_of, Daughter_of, etc.)
        
        Example queries should use:
        - toLower(p.name) CONTAINS toLower($name) for English names
        - toLower(p.name_chn) CONTAINS toLower($name) for Chinese names
        - r.association_type_desc for relationship type descriptions
        - Filter out kinship: NOT r.association_type_parent IN ['Kinship', 'Family', 'Blood relation']
        """
    
    def generate_cypher(self, question: str, context: Optional[str] = None) -> str:
        """Generate Cypher query for CBDB-specific questions"""
        
        # Debug: Show what question we're processing
        st.info(f"🤖 Text2Cypher: Processing question: '{question}'")
        
        system_prompt = f"""
        You are an expert in generating Cypher queries for the CBDB (China Biographical Database).
        
        {self.schema_context}
        
        Guidelines for CBDB queries:
        1. Always use case-insensitive matching for names: 
           - toLower(p.name) CONTAINS toLower($name) for English names
           - toLower(p.name_chn) CONTAINS toLower($name) for Chinese names
        2. Consider both English and Chinese names, surnames, and given names
        3. EXCLUDE kinship relationships in all relationship queries:
           - WHERE NOT r.association_type_parent IN ['Kinship', 'Family', 'Blood relation']
           - WHERE NOT toLower(r.association_desc) CONTAINS 'father' 
             AND NOT toLower(r.association_desc) CONTAINS 'mother'
             AND NOT toLower(r.association_desc) CONTAINS 'son'
             AND NOT toLower(r.association_desc) CONTAINS 'daughter'
        4. Return rich information including person_id, both name formats, and biographical details
        5. For relationships, include association metadata:
           - r.association_desc, r.association_type_desc, r.association_type_parent
        6. Use OPTIONAL MATCH for relationships that may not exist
        7. Limit results appropriately (usually 10-20 for person queries)
        
        Common query patterns:
        - Person lookup: MATCH (p:Person) WHERE (toLower(p.name) CONTAINS toLower($name) OR toLower(p.name_chn) CONTAINS toLower($name))
        - Non-kinship relationship queries: MATCH (p1:Person)-[r]-(p2:Person) WHERE NOT r.association_type_parent IN ['Kinship', 'Family', 'Blood relation'] RETURN r.association_desc, r.association_type_desc
        - Temporal queries: WHERE p.birth_year IS NOT NULL ORDER BY p.birth_year
        
        Return only the Cypher query, no explanations.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a Cypher query for: {question}"}
        ]
        
        if context:
            messages.insert(-1, {"role": "user", "content": f"Additional context: {context}"})
            st.info(f"📝 Additional context provided: {context}")
        
        try:
            st.info("🔄 Sending request to OpenAI...")
            
            # Try with the specified model first
            model_to_use = "gpt-5-2025-08-07"
            st.info(f"🤖 Using model: {model_to_use}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_completion_tokens=500
                )
            except Exception as model_error:
                st.warning(f"⚠️ Model {model_to_use} failed: {str(model_error)}")
                st.info("🔄 Trying fallback model: gpt-3.5-turbo")
                
                # Fallback to a standard model
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=500
                )
            
            cypher_query = response.choices[0].message.content.strip()
            st.info(f"📥 Raw response from OpenAI: {cypher_query}")
            
            # Clean up the query (remove markdown formatting if present)
            cleaned_query = re.sub(r'```cypher\s*\n?|```\s*\n?', '', cypher_query)
            cleaned_query = cleaned_query.strip()
            
            if cleaned_query != cypher_query:
                st.info(f"🧹 Cleaned query: {cleaned_query}")
            
            return cleaned_query
            
        except Exception as e:
            st.error(f"❌ Error generating Cypher query: {str(e)}")
            st.error(f"🔍 Exception details: {type(e).__name__}")
            return ""


class CBDBRetrievers:
    """Collection of specialized retrievers for CBDB data"""
    
    def __init__(self, neo4j_driver, text2cypher: CBDBText2Cypher):
        self.driver = neo4j_driver
        self.text2cypher = text2cypher
    
    def person_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Get detailed information about a person by name"""
        st.info(f"👤 person_by_name: Searching for '{name}'")
        
        query = """
        MATCH (p:Person)
        WHERE (toLower(p.name) CONTAINS toLower($name) 
           OR toLower(p.name_chn) CONTAINS toLower($name)
           OR toLower(p.surname) CONTAINS toLower($name)
           OR toLower(p.mingzi) CONTAINS toLower($name))
        RETURN p {
            .person_id, .name, .name_chn, .surname, .surname_chn,
            .mingzi, .mingzi_chn, .birth_year, .death_year, 
            .age_at_death, .index_year, .gender, .notes
        } AS person
        ORDER BY p.birth_year
        LIMIT 10
        """
        
        st.info(f"🔍 Executing person lookup query...")
        
        try:
            records, _, _ = self.driver.execute_query(query, name=name)
            results = [record.data() for record in records]
            st.success(f"✅ Found {len(results)} people matching '{name}'")
            
            if results:
                # Show sample result
                sample = results[0]['person']
                st.info(f"🔍 Sample: {sample.get('name', 'N/A')} ({sample.get('name_chn', 'N/A')})")
            
            return results
        except Exception as e:
            st.error(f"❌ Error in person_by_name: {str(e)}")
            return []
    
    def person_relationships(self, name: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get non-kinship relationships for a specific person"""
        base_query = """
        MATCH (p:Person)
        WHERE (toLower(p.name) CONTAINS toLower($name)
           OR toLower(p.name_chn) CONTAINS toLower($name))
        """
        
        # Non-kinship filter
        non_kinship_filter = """
        AND r.association_type_parent IS NOT NULL 
        AND NOT r.association_type_parent IN ['Kinship', 'Family', 'Blood relation']
        AND NOT toLower(r.association_desc) CONTAINS 'father' 
        AND NOT toLower(r.association_desc) CONTAINS 'mother'
        AND NOT toLower(r.association_desc) CONTAINS 'son'
        AND NOT toLower(r.association_desc) CONTAINS 'daughter'
        AND NOT toLower(r.association_desc) CONTAINS 'spouse'
        AND NOT toLower(r.association_desc) CONTAINS 'wife'
        AND NOT toLower(r.association_desc) CONTAINS 'husband'
        """
        
        if relationship_type:
            query = base_query + f"""
            MATCH (p)-[r:{relationship_type}]-(related:Person)
            WHERE 1=1
            {non_kinship_filter}
            RETURN p {{.person_id, .name, .name_chn}} AS person, 
                   type(r) AS relationship_type,
                   r {{.association_desc, .association_desc_chn, .association_type_desc, 
                       .association_type_parent, .association_code}} AS relationship_details,
                   related {{.person_id, .name, .name_chn, .birth_year, .death_year}} AS related_person
            ORDER BY related.birth_year
            LIMIT 20
            """
        else:
            query = base_query + f"""
            MATCH (p)-[r]-(related:Person)
            WHERE 1=1
            {non_kinship_filter}
            RETURN p {{.person_id, .name, .name_chn}} AS person, 
                   type(r) AS relationship_type,
                   r {{.association_desc, .association_desc_chn, .association_type_desc, 
                       .association_type_parent, .association_code}} AS relationship_details,
                   related {{.person_id, .name, .name_chn, .birth_year, .death_year}} AS related_person
            ORDER BY related.birth_year
            LIMIT 20
            """
        
        try:
            records, _, _ = self.driver.execute_query(query, name=name)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in person_relationships: {str(e)}")
            return []
    
    def people_by_period(self, period: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Get people from a specific period or search notes for period references"""
        # Search in notes for period references
        query = """
        MATCH (p:Person)
        WHERE any(note in p.notes WHERE toLower(note) CONTAINS toLower($period))
        RETURN p {
            .person_id, .name, .name_chn, .surname, .surname_chn,
            .birth_year, .death_year, .notes
        } AS person
        ORDER BY p.birth_year
        LIMIT $limit
        """
        try:
            records, _, _ = self.driver.execute_query(query, period=period, limit=limit)
            return [record.data() for record in records]
        except Exception as e:
            st.error(f"Error in people_by_period: {str(e)}")
            return []
    
    def people_by_time_period(self, start_year: int, end_year: int, limit: int = 15) -> List[Dict[str, Any]]:
        """Get people who lived during a specific time period"""
        query = """
        MATCH (p:Person)
        WHERE ((p.birth_year IS NOT NULL AND p.birth_year >= $start_year AND p.birth_year <= $end_year)
           OR (p.death_year IS NOT NULL AND p.death_year >= $start_year AND p.death_year <= $end_year)
           OR (p.birth_year IS NOT NULL AND p.death_year IS NOT NULL AND 
               p.birth_year <= $start_year AND p.death_year >= $end_year)
           OR (p.index_year IS NOT NULL AND p.index_year >= $start_year AND p.index_year <= $end_year))
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
        st.info(f"🎯 Starting text2cypher_query for: '{question}'")
        
        try:
            # Generate the Cypher query
            cypher_query = self.text2cypher.generate_cypher(question)
            
            if not cypher_query:
                st.warning("⚠️ No Cypher query was generated")
                return []
            
            st.success(f"✅ Generated Cypher: {cypher_query}")
            
            # Execute the query
            st.info("🚀 Executing Cypher query...")
            records, summary, keys = self.driver.execute_query(cypher_query)
            
            # Show execution statistics
            st.info(f"📊 Query executed successfully!")
            st.info(f"🔢 Records returned: {len(records)}")
            st.info(f"🔑 Result keys: {keys}")
            
            if hasattr(summary, 'counters'):
                st.info(f"📈 Database counters: {summary.counters}")
            
            # Process results
            results = [record.data() for record in records]
            
            # Show sample results for debugging
            if results:
                st.success(f"✅ Successfully processed {len(results)} results")
                st.info(f"🔍 Sample result: {str(results[0])[:200]}...")
            else:
                st.warning("⚠️ Query executed successfully but returned no results")
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error in text2cypher_query: {str(e)}")
            st.error(f"🔍 Exception type: {type(e).__name__}")
            
            # Try to show more details about the error
            if hasattr(e, 'code'):
                st.error(f"📋 Error code: {e.code}")
            if hasattr(e, 'message'):
                st.error(f"💬 Error message: {e.message}")
                
            return []
    
    def people_by_relationship_type(self, relationship_desc: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Find people connected by a specific non-kinship relationship type"""
        query = """
        MATCH (p1:Person)-[r]-(p2:Person)
        WHERE (toLower(r.association_desc) CONTAINS toLower($relationship_desc)
           OR toLower(r.association_type_desc) CONTAINS toLower($relationship_desc)
           OR toLower(r.association_type_parent) CONTAINS toLower($relationship_desc))
        AND r.association_type_parent IS NOT NULL 
        AND NOT r.association_type_parent IN ['Kinship', 'Family', 'Blood relation']
        AND NOT toLower(r.association_desc) CONTAINS 'father' 
        AND NOT toLower(r.association_desc) CONTAINS 'mother'
        AND NOT toLower(r.association_desc) CONTAINS 'son'
        AND NOT toLower(r.association_desc) CONTAINS 'daughter'
        AND NOT toLower(r.association_desc) CONTAINS 'spouse'
        AND NOT toLower(r.association_desc) CONTAINS 'wife'
        AND NOT toLower(r.association_desc) CONTAINS 'husband'
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
                        "description": "Get non-kinship relationships for a specific person (excludes family/blood relations)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The person's name",
                                },
                                "relationship_type": {
                                    "type": "string",
                                    "description": "Optional: specific non-kinship relationship type (e.g., 'writing_in_his_stead_done_by', 'student_of')",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                },
                "function": self.retrievers.person_relationships
            },
            "people_by_period": {
                "description": {
                    "type": "function",
                    "function": {
                        "name": "people_by_period",
                        "description": "Get people from specific periods or search biographical notes for period references",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "period": {
                                    "type": "string",
                                    "description": "Period name or term to search in biographical notes",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default 15)",
                                    "default": 15
                                }
                            },
                            "required": ["period"],
                        },
                    },
                },
                "function": self.retrievers.people_by_period
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
                        "description": "Use for complex queries about people and their non-kinship relationships. Generates custom Cypher queries",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The complex question about historical figures and their non-kinship relationships",
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
                        "description": "Find people connected by specific non-kinship relationships (e.g., 'writing', 'political', 'student', 'literary associate') - excludes family relationships",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "relationship_desc": {
                                    "type": "string",
                                    "description": "The non-kinship relationship type or description to search for",
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
        
        st.info(f"🧭 Router: Analyzing question: '{question}'")
        
        system_prompt = """
        You are an expert router for the CBDB (China Biographical Database) system.
        Your job is to choose the right tool to answer questions about historical figures 
        and their non-kinship relationships.
        
        Guidelines:
        - Use person_by_name for basic biographical information about specific people
        - Use person_relationships for questions about specific person's non-kinship connections
        - Use people_by_period for questions about specific periods or search biographical notes
        - Use people_by_time_period for questions about specific time periods or years
        - Use people_by_relationship_type for questions about specific non-kinship relationship types or categories
        - Use text2cypher_query for complex queries that require custom database searches
        - Use answer_from_context when the answer is already in the conversation
        
        IMPORTANT: All tools automatically exclude kinship relationships.
        Choose the most specific tool available for the question.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if context:
            messages.extend(context)
            st.info(f"📝 Router: Using context with {len(context)} messages")
        
        messages.append({
            "role": "user", 
            "content": f"Route this CBDB question to the appropriate tool: {question}"
        })
        
        try:
            st.info("🔄 Router: Sending request to OpenAI for tool selection...")
            
            # Try with the specified model first
            model_to_use = "gpt-5-2025-08-07"
            st.info(f"🤖 Router using model: {model_to_use}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    tools=[tool["description"] for tool in self.tools.values()],
                    tool_choice="auto",
                )
            except Exception as model_error:
                st.warning(f"⚠️ Router model {model_to_use} failed: {str(model_error)}")
                st.info("🔄 Router trying fallback model: gpt-3.5-turbo")
                
                # Fallback to a standard model
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    tools=[tool["description"] for tool in self.tools.values()],
                    tool_choice="auto",
                )
            
            if response.choices[0].message.tool_calls:
                st.info(f"✅ Router: Selected {len(response.choices[0].message.tool_calls)} tool(s)")
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    st.info(f"🔧 Tool {i+1}: {tool_call.function.name} with args: {tool_call.function.arguments}")
                return self._handle_tool_calls(response.choices[0].message.tool_calls)
            else:
                st.warning("⚠️ Router: No appropriate tool selected")
                return [{"error": "No appropriate tool selected"}]
                
        except Exception as e:
            st.error(f"❌ Router error: {str(e)}")
            st.error(f"🔍 Exception type: {type(e).__name__}")
            return [{"error": str(e)}]
    
    def _handle_tool_calls(self, tool_calls) -> List[Any]:
        """Execute the selected tools"""
        results = []
        
        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            st.info(f"🔧 Executing tool {i+1}/{len(tool_calls)}: {function_name}")
            st.info(f"📋 Arguments: {function_args}")
            
            if function_name in self.tools:
                try:
                    function_to_call = self.tools[function_name]["function"]
                    st.info(f"▶️ Calling {function_name}...")
                    result = function_to_call(**function_args)
                    
                    if isinstance(result, list):
                        st.success(f"✅ {function_name} returned {len(result)} results")
                        if result:
                            st.info(f"🔍 Sample result: {str(result[0])[:150]}...")
                    else:
                        st.success(f"✅ {function_name} completed")
                        st.info(f"🔍 Result type: {type(result).__name__}")
                    
                    results.append(result)
                except Exception as e:
                    error_msg = f"Error calling {function_name}: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    st.error(f"🔍 Exception type: {type(e).__name__}")
                    results.append({"error": error_msg})
            else:
                error_msg = f"Unknown function: {function_name}"
                st.error(f"❌ {error_msg}")
                results.append({"error": error_msg})
        
        st.info(f"🏁 Tool execution completed. Total results: {len(results)}")
        return results


class CBDBResponseSynthesizer:
    """Synthesizes comprehensive responses from CBDB data"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def synthesize_response(self, original_question: str, context: List[Dict[str, str]]) -> str:
        """Generate the final enhanced response"""
        
        st.info(f"📝 Synthesizer: Creating response for '{original_question}'")
        st.info(f"📚 Using {len(context)} context messages")
        
        system_prompt = """
        You are a knowledgeable assistant specializing in Chinese historical biographical data.
        
        Your task is to provide a comprehensive, well-structured answer based on the retrieved
        CBDB data. Your response should:
        
        1. Be historically accurate and based only on the provided data
        2. Include relevant biographical details (names, dates, offices, relationships)
        3. Explain relationships and historical context when available
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
            "content": f"Based on the retrieved information, provide a comprehensive answer to: {original_question}"
        })
        
        try:
            st.info("🔄 Synthesizer: Generating response with OpenAI...")
            st.info(f"📤 Sending {len(messages)} messages to OpenAI")
            
            # Log the context data being sent (truncated for readability)
            for i, msg in enumerate(messages):
                content_preview = str(msg.get('content', ''))[:200] + "..." if len(str(msg.get('content', ''))) > 200 else str(msg.get('content', ''))
                st.info(f"   Message {i+1} ({msg.get('role', 'unknown')}): {content_preview}")
            
            # Try with the specified model first
            model_to_use = "gpt-5-2025-08-07"
            st.info(f"🤖 Trying model: {model_to_use}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_completion_tokens=1500
                )
            except Exception as model_error:
                st.warning(f"⚠️ Model {model_to_use} failed: {str(model_error)}")
                st.info("🔄 Trying fallback model: gpt-3.5-turbo")
                
                # Fallback to a standard model
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=1500  # Note: different parameter name for older models
                )
            
            # Debug the response object
            st.info(f"📥 OpenAI response received")
            st.info(f"🔍 Response object type: {type(response)}")
            st.info(f"🔍 Number of choices: {len(response.choices) if hasattr(response, 'choices') else 'No choices attr'}")
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                st.info(f"🔍 Choice type: {type(choice)}")
                st.info(f"🔍 Choice has message: {hasattr(choice, 'message')}")
                
                if hasattr(choice, 'message'):
                    message = choice.message
                    st.info(f"🔍 Message type: {type(message)}")
                    st.info(f"🔍 Message has content: {hasattr(message, 'content')}")
                    st.info(f"🔍 Message content: {repr(message.content)}")
                    
                    final_response = message.content or ""  # Handle None content
                else:
                    final_response = ""
                    st.warning("⚠️ No message in choice")
            else:
                final_response = ""
                st.warning("⚠️ No choices in response")
            
            st.success(f"✅ Synthesizer: Generated {len(final_response)} character response")
            
            if not final_response:
                st.warning("⚠️ Empty response received from OpenAI!")
                st.info("🔍 This could be due to:")
                st.info("   - Content filtering")
                st.info("   - Model access issues")
                st.info("   - Invalid request format")
                st.info("   - API quota/rate limits")
                st.info("   - Data too large for processing")
                
                # Try to provide a basic fallback response
                st.info("🔄 Generating fallback response...")
                return "I apologize, but I was unable to generate a proper response from the retrieved data. Please check the debug information above for details about what data was found."
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(f"❌ Synthesizer error: {error_msg}")
            st.error(f"🔍 Exception type: {type(e).__name__}")
            
            # Try to get more specific error details
            if hasattr(e, 'response'):
                st.error(f"🔍 Error response: {e.response}")
            if hasattr(e, 'code'):
                st.error(f"🔍 Error code: {e.code}")
            if hasattr(e, 'status_code'):
                st.error(f"🔍 Status code: {e.status_code}")
                
            return error_msg


class CBDBAgenticRAG:
    """Simplified orchestrator for the CBDB multi-agent RAG system"""
    
    def __init__(self, neo4j_driver, openai_client):
        self.driver = neo4j_driver
        self.client = openai_client
        
        # Initialize components
        self.text2cypher = CBDBText2Cypher(neo4j_driver, openai_client)
        self.retrievers = CBDBRetrievers(neo4j_driver, self.text2cypher)
        self.router = CBDBRetrieverRouter(self.retrievers, openai_client)
        self.synthesizer = CBDBResponseSynthesizer(openai_client)
    
    def process_query(self, question: str) -> str:
        """Process a CBDB query through the simplified multi-agent system"""
        
        st.info(f"🚀 CBDB AgenticRAG: Starting to process query: '{question}'")
        
        # Route question and get results
        st.info("📍 Step 1: Routing question to appropriate retriever...")
        results = self.router.route_question(question)
        
        # Analyze results
        total_items = 0
        for result in results:
            if isinstance(result, list):
                total_items += len(result)
        
        st.info(f"📊 Step 2: Retrieved {len(results)} result set(s) containing {total_items} total items")
        
        # Show summary of results
        for i, result in enumerate(results):
            if isinstance(result, list):
                st.info(f"   Result set {i+1}: {len(result)} items")
            elif isinstance(result, dict) and "error" in result:
                st.warning(f"   Result set {i+1}: Error - {result['error']}")
            else:
                st.info(f"   Result set {i+1}: {type(result).__name__}")
        
        # Build context for response synthesis
        context = [{
            "role": "assistant",
            "content": f"Retrieved data for '{question}': {json.dumps(results, ensure_ascii=False, indent=2)}"
        }]
        
        # Generate final response
        st.info("📝 Step 3: Synthesizing final response...")
        response = self.synthesizer.synthesize_response(question, context)
        
        st.success(f"✅ Complete! Generated response of {len(response)} characters")
        
        return response
