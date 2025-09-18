# FAQ Retrieval System

An intelligent FAQ system that combines semantic search, keyword matching, and AI generation to provide accurate, contextual answers to user questions.

## 🎯 Problem Analysis

**Core Challenge**: Users ask questions in natural language with varying phrasing, requiring the system to match intent rather than exact words while handling ambiguous queries gracefully.

**Key Assumptions**:
- Users prefer quick, accurate answers over comprehensive but slow responses
- Confidence indication helps users understand response quality
- System should escalate to human support when appropriate
- Real-time feedback improves system performance over time

## 🏗️ Architecture & Design Decisions

### Why Hybrid Retrieval?

I chose a **hybrid approach** combining three strategies instead of relying on semantic search alone:

1. **Semantic Search** (60% weight): OpenAI embeddings for intent understanding
2. **Keyword Matching** (30% weight): Ensures important terms aren't missed
3. **Popularity Boosting** (10% weight): Learns from user interactions

**Alternative Considered**: Pure semantic search was simpler but missed exact keyword matches. Pure keyword search couldn't handle paraphrasing.

### Confidence-Based Response Strategy

| Confidence | Action | Rationale |
|------------|--------|-----------|
| > 0.8 | Direct FAQ Answer | High confidence, immediate response |
| 0.35-0.8 | FAQ + Context | Medium confidence, provide answer with context |
| 0.2-0.35 | AI Generated Response | Low confidence, use OpenAI for helpful guidance |
| < 0.2 | Escalate to Human | Very low confidence, human intervention needed |

**What I Decided NOT to Do**: 
- No complex ML training pipeline (chose OpenAI for reliability)
- No real-time retraining (chose feedback logging for future improvements)
- No complex user authentication (focused on core FAQ functionality)

### Data Quality Management

**Issues Found in Original Data**:
- Empty questions, nonsensical entries ("asdfghjkl", "What's the best pizza topping?")
- Mismatched Q&A pairs (pizza question with 2FA answer)
- Duplicate questions with slight variations

**Cleaning Strategy**: Pattern-based filtering, duplicate detection, automatic categorization

### Technology Stack

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| FastAPI | Web Framework | Async support, automatic API docs, type safety |
| OpenAI API | Embeddings & Generation | State-of-the-art semantic understanding |
| ArangoDB | Analytics & Persistence | Graph database for complex query analysis |
| Docker | Deployment | Consistent environments, easy scaling |

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key
- Ports 8000 and 8529 available

### Setup Instructions

1. **Clone and navigate**:
```bash
git clone <repository>
cd faq-rag
```

2. **Set OpenAI API key**:
```bash
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

3. **Start the system**:
```bash
./start.sh
```

The script will:
- Check dependencies and Docker status
- Build and start all services (FAQ app + ArangoDB)
- Wait for health checks to pass
- Open your browser automatically
- Display management commands

**That's it!** The system will be available at `http://localhost:8000`

## 📁 Project Structure

```
faq-rag/
├── app.py              # Main FastAPI application with hybrid retrieval logic
├── data.py             # FAQ database (cleaned from original messy data)
├── index.html          # Frontend interface with confidence visualization
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Multi-service orchestration
├── start.sh           # Automated startup script
└── test_db_integration.py  # Database integration tests
```

## 🔧 API Endpoints

### Main Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I change my password?"}'
```

**Response includes**: answer, confidence score, response type, processing time, alternative suggestions

### Other Endpoints
- `GET /health` - System health with database status
- `GET /stats` - Usage statistics and performance metrics  
- `GET /categories` - FAQ categories with counts
- `POST /feedback` - Submit response quality feedback

## 🧪 Testing the System

### Recommended Test Queries

**High Confidence** (should return direct FAQ):
- "How do I change my password?"
- "How can I update my billing information?"

**Medium Confidence** (FAQ with clarification):
- "I want to modify my login credentials"
- "Need help with account security settings"

**Low Confidence** (AI generated response):
- "I'm having trouble with my account"
- "Something is wrong with the app"

**Off-Topic** (helpful redirection):
- "What's the weather like?"
- "How do I cook pasta?"

### Database Integration Test
```bash
python3 test_db_integration.py
```

## 🔧 Management Commands

```bash
# View logs
docker-compose logs -f

# Stop system
docker-compose down

# Restart
docker-compose restart

# Check status
docker-compose ps
```

## 📊 System Intelligence

### What Makes It Smart

1. **Intent Detection**: Recognizes off-topic queries and provides helpful redirections
2. **Context Understanding**: Normalizes question variations ("How do I" vs "How can I")
3. **Graceful Degradation**: Falls back to AI generation when FAQ confidence is low
4. **Learning**: Tracks popularity scores and user feedback for continuous improvement

### Performance Features

- **Embedding Caching**: Avoids repeated OpenAI API calls
- **Async Processing**: Handles concurrent requests efficiently
- **Bulk Operations**: Efficient database operations with ArangoDB
- **Health Monitoring**: Comprehensive health checks and error handling

## 🚧 Limitations & Future Improvements

### Current Limitations

1. **No User Authentication**: All interactions are anonymous
2. **Static FAQ Database**: FAQs are loaded from static file, not admin interface
3. **Simple Feedback Loop**: Feedback is logged but not used for real-time improvements
4. **English Only**: No multi-language support
5. **Basic Analytics**: Limited to simple usage statistics

### With More Time, I Would Add

**Immediate Improvements**:
- Admin interface for FAQ management
- User authentication and personalization
- Advanced analytics dashboard with query pattern analysis
- A/B testing for different response strategies

**Advanced Features**:
- Multi-language support with language detection
- Voice interface integration
- Custom embedding fine-tuning based on feedback
- Integration with external knowledge bases
- Real-time collaborative FAQ editing

**Production Enhancements**:
- Rate limiting and API authentication
- Comprehensive monitoring and alerting
- Load balancing for high availability
- CDN integration for global performance

### Trade-offs Made

- **Simplicity over Complexity**: Chose OpenAI over custom ML models for faster development
- **Performance over Features**: Focused on core functionality rather than advanced features
- **Reliability over Innovation**: Used proven technologies instead of cutting-edge solutions

## 🛠️ Troubleshooting

**Port Conflicts**:
```bash
netstat -an | grep ':8000\|:8529'
sudo lsof -ti:8000 | xargs kill -9
```

**Docker Issues**:
```bash
docker system prune -a
# Restart Docker Desktop on Mac/Windows
```

**View Logs**:
```bash
docker-compose logs -f faq-app
docker-compose logs -f arangodb
```

---

**🤖 Ready to answer your questions!** Visit `http://localhost:8000` to start chatting.