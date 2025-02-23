import asyncio
from temporalio.client import Client

async def main():

    client = await Client.connect("localhost:7233")
    
    result = await client.execute_workflow(
        "EnsembleWorkflow",            
        "path/to/audio/file.wav",     
        id="audio_workflow",              
        task_queue="ensemble-task-queue", 
    )
    
    print("Workflow result:", result)

if __name__ == "__main__":
    asyncio.run(main())