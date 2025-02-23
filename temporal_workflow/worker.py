import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from workflow import EnsembleWorkflow, classify_audio

async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="ensemble-task-queue",
        workflows=[EnsembleWorkflow],
        activities=[classify_audio],
    )
    print("Worker started, waiting for tasks...")
    await worker.run() 

if __name__ == "__main__":
    asyncio.run(main())