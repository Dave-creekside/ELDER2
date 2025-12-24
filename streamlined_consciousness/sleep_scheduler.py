import asyncio
import logging
import time
from typing import Optional
from .consciousness_engine import StreamlinedConsciousness

logger = logging.getLogger("sleep-scheduler")

class SleepScheduler:
    """
    Background orchestrator that monitors trace collection and triggers Deep Sleep cycles.
    """
    
    def __init__(self, consciousness: StreamlinedConsciousness, 
                 check_interval: int = 60,
                 trace_threshold: int = 50,
                 nap_interval_seconds: int = 7200): # 2 hours
        self.consciousness = consciousness
        self.check_interval = check_interval
        self.trace_threshold = trace_threshold
        self.nap_interval = nap_interval_seconds
        self.last_sleep_time = time.time()
        self.running = False
        self._task = None

    async def start(self):
        """Start the background monitoring task"""
        if self.running:
            return
            
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"💤 Sleep Scheduler started (Threshold: {self.trace_threshold} traces)")

    async def stop(self):
        """Stop the background monitoring task"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("💤 Sleep Scheduler stopped")

    async def _run(self):
        """Main loop that polls Qdrant for trace counts"""
        while self.running:
            try:
                # Poll Qdrant for trace count (check if engine exists, don't trigger lazy load here)
                if self.consciousness.sleep_engine and self.consciousness.sleep_engine.qdrant:
                    await self.consciousness.sleep_engine._ensure_connections()
                    
                    # Count points in shadow_traces
                    collection_info = await self.consciousness.sleep_engine.qdrant.client.get_collection(
                        collection_name="shadow_traces"
                    )
                    count = collection_info.points_count
                    
                    logger.debug(f"Sleep Scheduler: {count}/{self.trace_threshold} traces gathered")
                    
                    time_since_sleep = time.time() - self.last_sleep_time
                    
                    if count >= self.trace_threshold:
                        logger.info(f"🚀 Trace threshold reached ({count}). Inducing Deep Sleep...")
                        await self.consciousness.perform_deep_sleep()
                        self.last_sleep_time = time.time()
                    elif count > 0 and time_since_sleep >= self.nap_interval:
                        logger.info(f"😴 Nap interval reached ({int(time_since_sleep/60)}m since last sleep). Inducing Nap...")
                        await self.consciousness.perform_deep_sleep()
                        self.last_sleep_time = time.time()
                    
            except Exception as e:
                logger.error(f"Sleep Scheduler error: {e}")
                
            await asyncio.sleep(self.check_interval)
