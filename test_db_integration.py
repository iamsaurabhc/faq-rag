#!/usr/bin/env python3
"""
Test script for ArangoDB integration
=====================================

This script verifies that the ArangoDB integration is working correctly
by testing database connection, FAQ storage, and retrieval operations.
"""

import asyncio
import requests
import json
from datetime import datetime

def test_health_endpoint():
    """Test the health endpoint for database status"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        data = response.json()
        
        print(f"✅ Health Status: {data.get('status')}")
        print(f"📊 System Ready: {data.get('system_ready')}")
        print(f"🗄️  Database Connected: {data.get('database_connected')}")
        print(f"🧠 Embeddings Ready: {data.get('faqs_have_embeddings')}")
        print(f"📚 Total FAQs: {data.get('total_faqs')}")
        
        return data.get('system_ready', False)
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_database_info():
    """Test the database info endpoint"""
    print("\n🗄️  Testing database info endpoint...")
    try:
        response = requests.get("http://localhost:8000/database-info")
        data = response.json()
        
        print(f"🔗 Connected: {data.get('connected')}")
        if data.get('connected'):
            print(f"📊 Database: {data.get('database')}")
            print("📋 Collections:")
            for name, info in data.get('collections', {}).items():
                if info.get('exists'):
                    print(f"  - {name}: {info.get('count')} records")
                else:
                    print(f"  - {name}: ❌ Error - {info.get('error')}")
        else:
            print(f"❌ Connection failed: {data.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Database info check failed: {e}")

def test_chat_functionality():
    """Test chat functionality to ensure FAQ retrieval works"""
    print("\n💬 Testing chat functionality...")
    
    test_queries = [
        "How do I change my password?",
        "How can I update my billing information?",
        "What's the weather like?",  # Off-topic
        "I need help with my account"  # Ambiguous
    ]
    
    for query in test_queries:
        try:
            print(f"\n📝 Query: '{query}'")
            response = requests.post(
                "http://localhost:8000/chat",
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                metadata = data.get('metadata', {})
                
                print(f"🎯 Confidence: {metadata.get('confidence_score', 0):.2f}")
                print(f"📂 Response Type: {metadata.get('response_type')}")
                print(f"⏱️  Processing Time: {metadata.get('processing_time_ms')}ms")
                print(f"🏷️  Category: {metadata.get('category')}")
                
                if metadata.get('matched_faq_id'):
                    print(f"📋 Matched FAQ ID: {metadata.get('matched_faq_id')}")
                
                # Show first 100 chars of answer
                answer = data.get('answer', '')
                preview = answer[:100] + "..." if len(answer) > 100 else answer
                print(f"💡 Answer Preview: {preview}")
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Chat test failed for '{query}': {e}")

def test_stats_endpoint():
    """Test the statistics endpoint"""
    print("\n📊 Testing statistics endpoint...")
    try:
        response = requests.get("http://localhost:8000/stats")
        data = response.json()
        
        print(f"📈 Total Queries: {data.get('total_queries', 0)}")
        print(f"📚 Total FAQs: {data.get('total_faqs', 0)}")
        print(f"🎯 Average Confidence: {data.get('average_confidence', 0):.2f}")
        
        # Show response type distribution
        response_types = data.get('response_types', {})
        if response_types:
            print("📊 Response Types:")
            for rtype, count in response_types.items():
                if count > 0:
                    print(f"  - {rtype}: {count}")
        
        # Show database analytics if available
        db_analytics = data.get('database_analytics', {})
        if db_analytics:
            print("🗄️  Database Analytics:")
            for key, value in db_analytics.items():
                print(f"  - {key}: {value}")
                
    except Exception as e:
        print(f"❌ Stats test failed: {e}")

def test_categories_endpoint():
    """Test the categories endpoint"""
    print("\n🏷️  Testing categories endpoint...")
    try:
        response = requests.get("http://localhost:8000/categories")
        data = response.json()
        
        print("📋 Available Categories:")
        for category in data:
            name = category.get('name', 'Unknown')
            count = category.get('count', 0)
            print(f"  - {name}: {count} FAQs")
            
    except Exception as e:
        print(f"❌ Categories test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting ArangoDB Integration Tests")
    print("=" * 50)
    
    # Test health first
    if not test_health_endpoint():
        print("\n❌ System not ready. Please ensure the application is running.")
        return
    
    # Wait a moment for system to be fully ready
    print("\n⏳ Waiting for system to be fully ready...")
    import time
    time.sleep(3)
    
    # Run all tests
    test_database_info()
    test_chat_functionality()
    test_stats_endpoint()
    test_categories_endpoint()
    
    print("\n" + "=" * 50)
    print("✅ ArangoDB Integration Tests Complete!")
    print("\n💡 Tips:")
    print("- Check the ArangoDB web interface at http://localhost:8529")
    print("- Use credentials: root / faq_system_2024")
    print("- Explore the 'faq_system' database and its collections")

if __name__ == "__main__":
    main() 