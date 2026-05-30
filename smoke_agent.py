import sys, os
sys.path.insert(0, 'part2')

api_key = os.environ.get('GROQ_API_KEY', '')
if not api_key:
    print('[SKIP] GROQ_API_KEY not set.')
    print('       Set it with:  $env:GROQ_API_KEY="gsk_..."')
    sys.exit(0)

from agent import RetentionAgent
agent = RetentionAgent(api_key=api_key)
print('Agent initialised with model:', agent.model)

ar = agent.run('What can you help me with?')
print('Response (first 200 chars):', ar.content[:200])
print('Tool calls:', len(ar.tool_trace))
print('Latency:', round(ar.total_latency_ms / 1000, 2), 's')
print()
print('-- Retention flow test --')
ar2 = agent.run('I have customer TC-004711 on the line, they want to cancel')
print('Tools called:', [t.tool_name for t in ar2.tool_trace])
print('Latency:', round(ar2.total_latency_ms / 1000, 2), 's')
print('Response preview:', ar2.content[:300])
