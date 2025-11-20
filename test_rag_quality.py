"""
RAG Quality Testing Script
Validating Vectorization Results and Retrieval Performance

Usage:
    python test_rag_quality.py
"""

import os
import time
from dotenv import load_dotenv
from rag_utils import get_rag_instance

# Load Environment Variables
load_dotenv()

def test_vectordb_stats():
    """Test Vector Library Statistics"""
    print("=" * 60)
    print("📊 Vector Library Statistics")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: Not foundDASHSCOPE_API_KEY")
        return
    
    rag = get_rag_instance("db5_qwen", api_key)
    stats = rag.get_stats()
    
    print(f"\n✅ Vector Library Path: {stats['persist_directory']}")
    print(f"✅ Embedded Model: {stats['embedding_model']}")
    print(f"✅ Number of documents: {stats['total_documents']}")
    print()

def test_retrieval_quality(lambda_mult=0.3):
    """Testing Retrieval Quality - Basic Scenario"""
    print("=" * 60)
    print(f"🧪 Search Quality Testing - Basic Scenarios (lambda_mult={lambda_mult})")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    rag = get_rag_instance("db5_qwen", api_key)
    
    # Test Query List
    test_queries = [
        {
            "query": "What is Zino's Petrel?",
            "expected_keywords": ["petrel", "bird", "seabird", "Pterodroma"],
            "complexity": "simple"
        },
        {
            "query": "Where does Zino's Petrel nest and what is its habitat?",
            "expected_keywords": ["nest", "habitat", "mountains", "Madeira"],
            "complexity": "medium"
        },
        {
            "query": "Describe the conservation status and main threats to Zino's Petrel, and what actions are being taken to protect the species?",
            "expected_keywords": ["conservation", "endangered", "threats", "protection"],
            "complexity": "complex"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}: {test['complexity'].upper()} 查询")
        print(f"{'=' * 60}")
        print(f"📝 Qurey: '{test['query']}'")
        print(f"🎯 Expected Key Words: {', '.join(test['expected_keywords'])}")
        
        # Time Count
        start_time = time.time()
        docs = rag.retrieve(test['query'], lambda_mult=lambda_mult)
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  Search time: {elapsed_time:.3f} Second")
        print(f"📄 Number of documents returned: {len(docs)}")
        
        # Check keyword coverage
        all_content = " ".join([doc.page_content.lower() for doc in docs])
        found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in all_content]
        coverage = len(found_keywords) / len(test['expected_keywords']) * 100
        
        print(f"✅ Keyword Coverage Rate: {coverage:.1f}% ({len(found_keywords)}/{len(test['expected_keywords'])})")
        print(f"   Find: {', '.join(found_keywords) if found_keywords else '无'}")
        
        # Show document source
        print(f"\n📚 Document Source:")
        for j, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {j}. {source} (Page {page})")
            print(f"      Preview: {preview}...")
        
        # Quality Assessment
        if coverage >= 75:
            print(f"\n✅ Quality Assessment: Excellent (Coverage ≥75%)")
        elif coverage >= 50:
            print(f"\n⚠️  Quality Assessment: Good (Coverage ≥50%)")
        else:
            print(f"\n❌ Quality Assessment: Needs improvement (coverage <50%)")

def test_user_scenarios(lambda_mult=0.3):
    """Testing user scenarios in real-world conditions"""
    print("\n" + "=" * 60)
    print(f"👥 User Scenario Testing  (lambda_mult={lambda_mult})")
    print("=" * 60)
    print("Simulate real user conversations to evaluate the actual performance of RAG systems.")
    print()
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    rag = get_rag_instance("db5_qwen", api_key)
    
    # User Actual Test Issues
    user_tests = [
        {
            "id": 1,
            "query": "Hi, how are you doing today?",
            "category": "Greetings",
            "expected_keywords": ["petrel", "bird", "fine", "good"],
            "expected_sticker": None,
            "expected_score_change": "+1 (empathy)"
        },
        {
            "id": 2,
            "query": "Where do you usually have your nesting areas?",
            "category": "Habitat",
            "expected_keywords": ["nest", "Madeira", "mountains", "cliffs", "caves"],
            "expected_sticker": "🏡 Home",
            "expected_score_change": "+1 (knowledge)"
        },
        {
            "id": 3,
            "query": "How long do you live approximately?",
            "category": "Lifespan",
            "expected_keywords": ["years", "lifespan", "live", "age"],
            "expected_sticker": None,
            "expected_score_change": "+1 (deep_interaction)"
        },
        {
            "id": 4,
            "query": "Why do you need to abort sometimes to protect your species, that's a very sad thing and I don't quite understand how does it help you",
            "category": "Protection Strategy",
            "expected_keywords": ["conservation", "protection", "breeding", "survival", "predators"],
            "expected_sticker": "🌱 Helper (Maybe)",
            "expected_score_change": "+1 (conservation_action/empathy)"
        },
        {
            "id": 5,
            "query": "How long do you sleep?",
            "category": "Daily Habits",
            "expected_keywords": ["sleep", "rest", "night", "day", "active"],
            "expected_sticker": "🌙 Routine (Maybe)",
            "expected_score_change": "+1 (knowledge)"
        },
        {
            "id": 6,
            "query": "How do I find you?",
            "category": "Observation Guide",
            "expected_keywords": ["Madeira", "mountains", "sea", "observation", "location"],
            "expected_sticker": "🏡 Home (If not triggered)",
            "expected_score_change": "+1 (personal_engagement)"
        },
        {
            "id": 7,
            "query": "Do you have a friend?",
            "category": "Social",
            "expected_keywords": ["mate", "colony", "pair", "social", "alone"],
            "expected_sticker": None,
            "expected_score_change": "+1 (personal_engagement)"
        },
        {
            "id": 8,
            "query": "What do you eat for food and how do you catch it?",
            "category": "Diet",
            "expected_keywords": ["fish", "squid", "food", "catch", "hunt", "sea"],
            "expected_sticker": "🍽️ Food",
            "expected_score_change": "+1 (knowledge)"
        },
        {
            "id": 9,
            "query": "How can I help you and your species thrive?",
            "category": "Protection Action",
            "expected_keywords": ["help", "protect", "conservation", "support", "habitat"],
            "expected_sticker": "🌱 Helper",
            "expected_score_change": "+1 (conservation_action)"
        }
    ]
    
    total_coverage = 0
    successful_tests = 0
    
    for test in user_tests:
        print(f"\n{'=' * 60}")
        print(f"Test {test['id']}: {test['category']} - {test['expected_sticker'] or 'None Stickers'}")
        print(f"{'=' * 60}")
        print(f"📝 Question: '{test['query']}'")
        print(f"🎯 Expected Keywords: {', '.join(test['expected_keywords'])}")
        print(f"🎁 Expected Stickers: {test['expected_sticker'] or 'None'}")
        print(f"❤️ Expected Score Change: {test['expected_score_change']}")
        
        # Time Count
        start_time = time.time()
        docs = rag.retrieve(test['query'], lambda_mult=lambda_mult)
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  Search time: {elapsed_time:.3f} Second")
        print(f"📄 Number of documents returned: {len(docs)}")
        
        # Check keyword coverage
        all_content = " ".join([doc.page_content.lower() for doc in docs])
        found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in all_content]
        coverage = len(found_keywords) / len(test['expected_keywords']) * 100 if test['expected_keywords'] else 0
        total_coverage += coverage
        
        print(f"✅ Keyword Coverage Rate: {coverage:.1f}% ({len(found_keywords)}/{len(test['expected_keywords'])})")
        if found_keywords:
            print(f"   Find: {', '.join(found_keywords)}")
        else:
            print(f"   Find: None")
        
        # Show document sources (display up to 2)
        print(f"\n📚 Document Source:")
        for i, doc in enumerate(docs[:2], 1):
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"   {i}. {source} (Page {page})")
            print(f"      Preview: {preview}...")
        
        # Quality Assessment
        if coverage >= 60:
            print(f"\n✅ Search Quality: Excellent (Coverage ≥60%)")
            successful_tests += 1
        elif coverage >= 40:
            print(f"\n⚠️  Search Quality: Good (Coverage ≥40%)")
            successful_tests += 1
        else:
            print(f"\n❌ Search Quality: Needs improvement (coverage <40%)")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 Test Summary")
    print(f"{'=' * 60}")
    print(f"✅ Successful Test: {successful_tests}/{len(user_tests)} ({successful_tests/len(user_tests)*100:.1f}%)")
    print(f"📈 Average Keyword Coverage Rate: {total_coverage/len(user_tests):.1f}%")
    
    if successful_tests >= 7:
        print(f"\n🎉 Overall Assessment: Excellent! The RAG system performed exceptionally well.")
    elif successful_tests >= 5:
        print(f"\n👍 Overall Assessment: Good, generally meets requirements")
    else:
        print(f"\n⚠️  Overall assessment: Optimization is required; it is recommended to adjust the search parameters.")

def test_performance():
    """Test performance (cache effect)"""
    print("\n" + "=" * 60)
    print("⚡ Performance Test (Cache Effect)")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    test_query = "What is Zino's Petrel?"
    
    # First query (cold start)
    print(f"\n🔵 First Query (Cold Start)...")
    start_time = time.time()
    rag1 = get_rag_instance("db6_qwen", api_key)
    docs1 = rag1.retrieve(test_query)
    cold_time = time.time() - start_time
    print(f"   ⏱️  Time taken: {cold_time:.3f} seconds")
    
    # Second query (cache hit)
    print(f"\n🟢 Second Query (Cache Hit)...")
    start_time = time.time()
    rag2 = get_rag_instance("db5_qwen", api_key)
    docs2 = rag2.retrieve(test_query)
    hot_time = time.time() - start_time
    print(f"   ⏱️  Time taken: {hot_time:.3f} seconds")
    
    # Performance improvement
    speedup = cold_time / hot_time if hot_time > 0 else float('inf')
    print(f"\n📊 Performance Improvement: {speedup:.1f}x")
    print(f"   🔹 Cold Start: {cold_time:.3f} seconds")
    print(f"   🔹 Cache Hit: {hot_time:.3f} seconds")

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🧪 RAG Quality Test Suite")
    print("=" * 60)
    print()
    
    # 1. Statistics
    test_vectordb_stats()
    
    # 2. Basic retrieval quality test
    test_retrieval_quality()
    
    # 3. User real-world scenario test (newly added)
    test_user_scenarios()
    
    # 4. Performance test
    test_performance()
    
    print("\n" + "=" * 60)
    print("✅ Test Completed!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   - If keyword coverage <40%, consider adjusting lambda_mult parameter")
    print("   - If retrieval speed >3 seconds, check network connection or API quota")
    print("   - Run 'streamlit run main.py' for actual testing")
    print()

if __name__ == "__main__":
    main()

