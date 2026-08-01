import asyncio
from backend.state import make_initial_state
from backend.orchestration.graph import graph_app
async def run():
    state = make_initial_state('Who is the manager of engineering?', 'test1234')
    config = {'configurable': {'thread_id': 'test1234'}}
    
    # We must also ensure AgentFactory is initialized
    from backend.orchestration.factory import AgentFactory
    await AgentFactory.initialize()

    async for event in graph_app.astream_events(state, config=config, version='v2'):
        if event['event'] == 'on_chat_model_stream':
            node_name = event.get('metadata', {}).get('langgraph_node')
            if node_name == 'combiner':
                print(f'COMBINER CHUNK: {repr(event["data"]["chunk"].content)}')

asyncio.run(run())
