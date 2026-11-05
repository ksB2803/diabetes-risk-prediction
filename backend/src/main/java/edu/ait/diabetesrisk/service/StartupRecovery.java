package edu.ait.diabetesrisk.service;

import edu.ait.diabetesrisk.entity.JobStatus;
import edu.ait.diabetesrisk.entity.TrainingJob;
import edu.ait.diabetesrisk.repository.TrainingJobRepository;
import java.time.Instant;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

/**
 * Jobs that were queued or running when the server stopped cannot resume
 * (the worker queue lives in memory), so they are marked as failed with a clear message.
 */
@Component
public class StartupRecovery implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(StartupRecovery.class);

    private final TrainingJobRepository jobs;

    public StartupRecovery(TrainingJobRepository jobs) {
        this.jobs = jobs;
    }

    @Override
    public void run(ApplicationArguments args) {
        List<TrainingJob> stale = jobs.findByStatusIn(List.of(JobStatus.QUEUED, JobStatus.RUNNING));
        for (TrainingJob job : stale) {
            job.setStatus(JobStatus.FAILED);
            job.setStage(TrainingWorker.STAGE_FAILED);
            job.setErrorMessage("The server restarted before this job finished. Start a new training job.");
            job.setFinishedAt(Instant.now());
            jobs.save(job);
        }
        if (!stale.isEmpty()) {
            log.info("Marked {} interrupted training job(s) as failed", stale.size());
        }
    }
}
