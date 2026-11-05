package edu.ait.diabetesrisk.config;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * A fixed-size worker pool with a bounded queue for training jobs.
 * When the queue is full, submissions are rejected (mapped to HTTP 429)
 * instead of spawning more threads.
 */
@Configuration
public class TrainingExecutorConfig {

    @Bean(name = "trainingExecutor", destroyMethod = "shutdownNow")
    public ThreadPoolExecutor trainingExecutor(AppProperties props) {
        int workers = Math.max(1, props.getTraining().getMaxConcurrentJobs());
        int capacity = Math.max(1, props.getTraining().getQueueCapacity());
        AtomicInteger counter = new AtomicInteger();
        return new ThreadPoolExecutor(
                workers,
                workers,
                0L,
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(capacity),
                runnable -> {
                    Thread t = new Thread(runnable, "training-worker-" + counter.incrementAndGet());
                    t.setDaemon(true);
                    return t;
                },
                new ThreadPoolExecutor.AbortPolicy());
    }
}
