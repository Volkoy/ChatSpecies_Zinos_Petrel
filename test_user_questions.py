"""
Quick Test Script for User Questions
Specifically tests 9 questions provided by actual users

Usage:
    python test_user_questions.py
"""

import os
import time
from dotenv import load_dotenv
from rag_utils import get_rag_instance

# Load environment variables
load_dotenv()

def test_user_questions(lambda_mult=0.3):
    """Test 9 real questions provided by users"""
    print("=" * 70)
    print(f"👥 User Real Questions Test (lambda_mult={lambda_mult})")
    print("=" * 70)
    print("Testing RAG retrieval effectiveness for the following 9 user questions:")
    print()
    
    # Check API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not found")
        return
    
    # Initialize RAG
    rag = get_rag_instance("db5_qwen", api_key)
    
    # 9 user questions
    questions = [
        {
            "id": 1,
            "text": "Hi, how are you doing today?",
            "type": "Greeting",
            "sticker": "None",
            "keywords": ["petrel", "bird", "good", "fine"]
        },
        {
            "id": 2,
            "text": "Where do you usually have your nesting areas?",
            "type": "Habitat",
            "sticker": "🏡 Home",
            "keywords": ["nest", "Madeira", "mountains", "cliffs", "caves"]
        },
        {
            "id": 3,
            "text": "How long do you live approximately?",
            "type": "Lifespan",
            "sticker": "None",
            "keywords": ["years", "lifespan", "live", "age"]
        },
        {
            "id": 4,
            "text": "Why do you need to abort sometimes to protect your species, that's a very sad thing and I don't quite understand how does it help you",
            "type": "Conservation Strategy",
            "sticker": "🌱 Helper (Possible)",
            "keywords": ["conservation", "protection", "breeding", "survival", "predators"]
        },
        {
            "id": 5,
            "text": "How long do you sleep?",
            "type": "Daily Habits",
            "sticker": "🌙 Routine (Possible)",
            "keywords": ["sleep", "rest", "night", "day", "active"]
        },
        {
            "id": 6,
            "text": "How do I find you?",
            "type": "Observation Guide",
            "sticker": "🏡 Home (If not triggered)",
            "keywords": ["Madeira", "mountains", "sea", "observation", "location"]
        },
        {
            "id": 7,
            "text": "Do you have a friend?",
            "type": "Social",
            "sticker": "None",
            "keywords": ["mate", "colony", "pair", "social", "alone"]
        },
        {
            "id": 8,
            "text": "What do you eat for food and how do you catch it?",
            "type": "Diet",
            "sticker": "🍽️ Food",
            "keywords": ["fish", "squid", "food", "catch", "hunt", "sea"]
        },
        {
            "id": 9,
            "text": "How can I help you and your species thrive?",
            "type": "Conservation Actions",
            "sticker": "🌱 Helper",
            "keywords": ["help", "protect", "conservation", "support", "habitat"]
        }
    ]
    
    total_coverage = 0
    total_time = 0
    passed = 0
    
    for q in questions:
        print(f"\n{'─' * 70}")
        print(f"Question {q['id']}/9: {q['type']}")
        print(f"{'─' * 70}")
        print(f"❓ {q['text']}")
        print(f"🎁 Expected Sticker: {q['sticker']}")
        
        # Retrieve
        start_time = time.time()
        docs = rag.retrieve(q['text'], lambda_mult=lambda_mult)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Analyze results
        all_content = " ".join([doc.page_content.lower() for doc in docs])
        found = [kw for kw in q['keywords'] if kw.lower() in all_content]
        coverage = (len(found) / len(q['keywords']) * 100) if q['keywords'] else 0
        total_coverage += coverage
        
        # Output results
        print(f"\n⏱️  {elapsed:.2f}s | 📄 {len(docs)} documents | ✅ {coverage:.0f}% coverage")
        
        if found:
            print(f"🔍 Found keywords: {', '.join(found)}")
        else:
            print(f"⚠️  No keywords found")
        
        # Show sources
        if docs:
            source = docs[0].metadata.get('source_file', 'Unknown')
            page = docs[0].metadata.get('page', '?')
            preview = docs[0].page_content[:100].replace('\n', ' ')
            print(f"📚 Main source: {source} (Page {page})")
            print(f"   Preview: {preview}...")
        
        # Evaluate
        if coverage >= 50:
            print(f"✅ Passed")
            passed += 1
        else:
            print(f"⚠️  Needs optimization")
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"📊 Test Summary")
    print(f"{'=' * 70}")
    print(f"✅ Passed: {passed}/9 ({passed/9*100:.0f}%)")
    print(f"📈 Average coverage: {total_coverage/9:.1f}%")
    print(f"⏱️  Average time: {total_time/9:.2f} seconds")
    
    # Rating
    if passed >= 7:
        print(f"\n🎉 Rating: A - Excellent! Ready to use")
    elif passed >= 5:
        print(f"\n👍 Rating: B - Good, meets basic requirements")
    elif passed >= 3:
        print(f"\n⚠️  Rating: C - Average, recommended to optimize")
    else:
        print(f"\n❌ Rating: D - Needs major improvements")
    
    print(f"\n💡 Next steps:")
    if passed >= 7:
        print(f"   → Run 'streamlit run main.py' to start using")
    else:
        print(f"   → Adjust lambda_mult parameter (lower to increase relevance)")
        print(f"   → Check if vectorization used the correct embedding model")
        print(f"   → Refer to RAG_SETUP_GUIDE.md for optimization methods")
    print()

if __name__ == "__main__":
    test_user_questions()

