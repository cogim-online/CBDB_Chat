"""
Simple Cypher Test Script

This script focuses specifically on testing the Cypher generation and execution
to help debug why you're getting no information from the CBDB system.
"""

import os
from neo4j import GraphDatabase
import openai

# Simple test class for Cypher functionality
class SimpleCypherTest:
    def __init__(self):
        self.driver = None
        self.openai_client = None

    def connect(self):
        """Setup connections with prompts"""
        # OpenAI setup
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = input("Enter OpenAI API Key: ")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI connected")

        # Neo4j setup
        uri = os.getenv('NEO4J_URI') or input("Neo4j URI: ")
        username = os.getenv('NEO4J_USERNAME') or "neo4j"
        password = os.getenv('NEO4J_PASSWORD') or input("Neo4j Password: ")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            print("✅ Neo4j connected")
            return True
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            return False

    def test_data_exists(self):
        """Check if data exists in database"""
        print("\n🔍 Checking database content...")
        
        queries = [
            ("Person count", "MATCH (p:Person) RETURN count(p) as count"),
            ("Sample persons", "MATCH (p:Person) RETURN p.name, p.name_chn, p.birth_year LIMIT 5"),
            ("Relationship count", "MATCH ()-[r]->() RETURN count(r) as count"),
            ("Relationship types", "MATCH ()-[r]->() RETURN type(r), count(*) as count ORDER BY count DESC LIMIT 5")
        ]
        
        for name, query in queries:
            try:
                records, _, _ = self.driver.execute_query(query)
                print(f"✅ {name}:")
                for record in records:
                    print(f"   {record.data()}")
            except Exception as e:
                print(f"❌ {name} failed: {e}")

    def generate_simple_cypher(self, question):
        """Generate Cypher using simplified schema"""
        schema = """
        Database Schema:
        - Person nodes with properties: name, name_chn, birth_year, death_year
        - Relationships between persons (exclude kinship)
        - Use CONTAINS for name matching: WHERE toLower(p.name) CONTAINS toLower($name)
        """
        
        prompt = f"""
        Given this Neo4j schema:
        {schema}
        
        Generate a Cypher query for: {question}
        
        Return only the Cypher query, no explanations.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using standard model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            cypher = response.choices[0].message.content.strip()
            # Remove markdown formatting
            cypher = cypher.replace('```cypher', '').replace('```', '').strip()
            return cypher
        except Exception as e:
            print(f"❌ Cypher generation failed: {e}")
            return None

    def test_queries(self):
        """Test various queries"""
        print("\n🧪 Testing queries...")
        
        test_cases = [
            "Find a person named Wang",
            "Show me 3 people from the database",
            "Find people born after 1000",
            "Show relationships between people"
        ]
        
        for question in test_cases:
            print(f"\n📝 Question: {question}")
            
            # Generate Cypher
            cypher = self.generate_simple_cypher(question)
            if not cypher:
                continue
                
            print(f"🔍 Generated: {cypher}")
            
            # Execute query
            try:
                records, _, _ = self.driver.execute_query(cypher)
                print(f"✅ Results: {len(records)} records")
                
                # Show first few results
                for i, record in enumerate(records[:2]):
                    print(f"   {i+1}: {record.data()}")
                    
            except Exception as e:
                print(f"❌ Execution failed: {e}")

    def test_manual_queries(self):
        """Test manually written queries"""
        print("\n🔧 Testing manual queries...")
        
        manual_queries = [
            ("Count all persons", "MATCH (p:Person) RETURN count(p) as total"),
            ("First 3 persons", "MATCH (p:Person) RETURN p.name, p.name_chn LIMIT 3"),
            ("People with names", "MATCH (p:Person) WHERE p.name IS NOT NULL RETURN p.name LIMIT 5"),
            ("Birth years", "MATCH (p:Person) WHERE p.birth_year IS NOT NULL RETURN p.name, p.birth_year ORDER BY p.birth_year LIMIT 5")
        ]
        
        for name, query in manual_queries:
            print(f"\n{name}:")
            print(f"Query: {query}")
            try:
                records, _, _ = self.driver.execute_query(query)
                print(f"✅ {len(records)} results:")
                for record in records:
                    print(f"   {record.data()}")
            except Exception as e:
                print(f"❌ Failed: {e}")

    def interactive_test(self):
        """Interactive testing mode"""
        print("\n💬 Interactive mode (type 'quit' to exit)")
        
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            # Generate and test Cypher
            cypher = self.generate_simple_cypher(question)
            if cypher:
                print(f"Generated Cypher: {cypher}")
                
                try:
                    records, _, _ = self.driver.execute_query(cypher)
                    print(f"Found {len(records)} results:")
                    for i, record in enumerate(records[:3]):
                        print(f"  {i+1}: {record.data()}")
                except Exception as e:
                    print(f"Execution error: {e}")

    def run(self):
        """Run all tests"""
        print("🚀 Simple Cypher Test Starting...")
        
        if not self.connect():
            return
            
        self.test_data_exists()
        self.test_manual_queries()
        self.test_queries()
        
        # Ask if user wants interactive mode
        if input("\nRun interactive mode? (y/n): ").lower().startswith('y'):
            self.interactive_test()
        
        if self.driver:
            self.driver.close()
        print("\n✅ Tests completed")

if __name__ == "__main__":
    tester = SimpleCypherTest()
    tester.run()
