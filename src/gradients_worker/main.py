import asyncio

from gradients_worker import constants as cst
from gradients_worker.finetune_scheduler import GradientsTrainingScheduler
from gradients_worker.logger import logger, task_context
from gradients_worker.utils import load_config


async def launch_gradients_training_scheduler(task_name: str):
    """Run periodic training with metrics collection from Gradients API."""
    with task_context(task_name):
        logger.info(f"Starting Gradients Metrics Worker for {task_name}")

        try:
            scheduler = GradientsTrainingScheduler(task_name)
            await scheduler.run_forever()
        except Exception as e:
            logger.error(f"Fatal error in training scheduler: {e}", exc_info=True)


async def main():
    """Run all periodic training tasks concurrently."""
    logger.info("Starting all training workers")

    config = load_config(cst.CONFIG_FILENAME)
    tasks = []

    for task_name, task_config in config.items():
        if not isinstance(task_config, dict) or not task_config.get(cst.KEY_ENABLED, False):
            continue
        tasks.append(launch_gradients_training_scheduler(task_name=task_name))

    if not tasks:
        logger.warning("No tasks are enabled in the configuration!")
        return

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
