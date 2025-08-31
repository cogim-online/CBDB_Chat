"""
Debug script for the 0-character response issue

This script will help debug why the synthesizer is returning empty responses.
"""

import os
import json
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_openai_response():
    """Test OpenAI API directly to see if it's working"""
    print("🧪 Testing OpenAI API directly...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Test simple request first
    simple_test = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50
    }
    
    try:
        print("🔄 Testing simple request...")
        response = client.chat.completions.create(**simple_test)
        content = response.choices[0].message.content
        print(f"✅ Simple test: '{content}' ({len(content)} chars)")
    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        return False
    
    # Test with the problematic model
    try:
        print("🔄 Testing gpt-5-mini-2025-08-07...")
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=50
        )
        content = response.choices[0].message.content
        print(f"✅ GPT-5-mini test: '{content}' ({len(content)} chars)")
    except Exception as e:
        print(f"❌ GPT-5-mini test failed: {e}")
        print("This could be why you're getting 0-character responses!")
        return False
    
    return True

def test_synthesizer_scenario():
    """Test a scenario similar to what the synthesizer receives"""
    print("\n🧪 Testing synthesizer scenario...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Simulate what the synthesizer gets
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
    
    # Sample data that might be retrieved
    sample_data = [
        {"person": {"name": "Wang Wei", "name_chn": "王維", "birth_year": 1020, "death_year": 1080}}
    ]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"Retrieved data for 'Who is Wang Wei?': {json.dumps(sample_data, ensure_ascii=False, indent=2)}"},
        {"role": "user", "content": "Based on the retrieved information, provide a comprehensive answer to: Who is Wang Wei?"}
    ]
    
    print(f"📤 Sending {len(messages)} messages")
    for i, msg in enumerate(messages):
        content_preview = str(msg['content'])[:100] + "..." if len(str(msg['content'])) > 100 else str(msg['content'])
        print(f"   Message {i+1} ({msg['role']}): {content_preview}")
    
    # Test with fallback model
    try:
        print("🔄 Testing with gpt-3.5-turbo...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        print(f"✅ GPT-3.5-turbo response: {len(content)} characters")
        if content:
            print(f"📝 Preview: {content[:200]}...")
        else:
            print("❌ Empty response from GPT-3.5-turbo!")
            
    except Exception as e:
        print(f"❌ GPT-3.5-turbo failed: {e}")
    
    # Test with the problematic model
    try:
        print("🔄 Testing with gpt-5-mini-2025-08-07...")
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=messages,
            max_completion_tokens=1500
        )
        content = response.choices[0].message.content
        print(f"✅ GPT-5-mini response: {len(content)} characters")
        if content:
            print(f"📝 Preview: {content[:200]}...")
        else:
            print("❌ Empty response from GPT-5-mini!")
            print("🔍 This is likely the source of your problem!")
            
    except Exception as e:
        print(f"❌ GPT-5-mini failed: {e}")
        print("🔍 This model access issue is likely causing the 0-character responses!")

def test_data_size_issue():
    """Test if large data causes issues"""
    print("\n🧪 Testing large data scenario...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Create a large data payload
    large_data = []
    for i in range(50):
        large_data.append({
            "person": {
                "name": f"Person {i}",
                "name_chn": f"人物{i}",
                "birth_year": 1000 + i,
                "death_year": 1050 + i,
                "notes": ["This is a long biographical note " * 10]
            }
        })
    
    messages = [
        {"role": "system", "content": "Summarize this data briefly."},
        {"role": "assistant", "content": f"Data: {json.dumps(large_data, ensure_ascii=False)}"},
        {"role": "user", "content": "Provide a summary of this historical data."}
    ]
    
    total_chars = sum(len(str(msg['content'])) for msg in messages)
    print(f"📊 Total message size: {total_chars} characters")
    
    if total_chars > 100000:  # ~100KB
        print("⚠️ Data might be too large for processing")
        print("🔍 This could cause empty responses due to token limits")
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500
        )
        content = response.choices[0].message.content
        print(f"✅ Large data test: {len(content)} characters")
        
    except Exception as e:
        print(f"❌ Large data test failed: {e}")
        if "too long" in str(e).lower() or "token" in str(e).lower():
            print("🔍 Token limit exceeded - this is likely your issue!")

def main():
    print("🔍 Debugging 0-Character Response Issue")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    # Run tests
    if test_openai_response():
        test_synthesizer_scenario()
        test_data_size_issue()
    
    print("\n" + "=" * 50)
    print("🏁 Debug complete")
    
    print("\n🔍 Common causes of 0-character responses:")
    print("1. Model 'gpt-5-mini-2025-08-07' not available/accessible")
    print("2. Content filtering blocking the response")
    print("3. Data payload too large (token limit exceeded)")
    print("4. API quota/rate limits reached")
    print("5. Invalid request format")

if __name__ == "__main__":
    main()
