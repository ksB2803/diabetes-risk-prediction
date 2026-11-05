package edu.ait.diabetesrisk.controller;

import edu.ait.diabetesrisk.client.MlServiceClient;
import edu.ait.diabetesrisk.client.MlServiceException;
import edu.ait.diabetesrisk.client.MlServiceUnavailableException;
import edu.ait.diabetesrisk.dto.DashboardSummaryDto;
import edu.ait.diabetesrisk.dto.HealthDto;
import edu.ait.diabetesrisk.entity.JobStatus;
import edu.ait.diabetesrisk.repository.PredictionHistoryRepository;
import edu.ait.diabetesrisk.repository.TrainedModelRepository;
import edu.ait.diabetesrisk.repository.TrainingJobRepository;
import java.time.Instant;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class HealthController {

    private final MlServiceClient ml;
    private final TrainingJobRepository jobs;
    private final TrainedModelRepository models;
    private final PredictionHistoryRepository predictions;

    public HealthController(MlServiceClient ml, TrainingJobRepository jobs, TrainedModelRepository models,
                            PredictionHistoryRepository predictions) {
        this.ml = ml;
        this.jobs = jobs;
        this.models = models;
        this.predictions = predictions;
    }

    /** Always 200 so the UI can show which part is down. */
    @GetMapping("/health")
    public HealthDto health() {
        String database;
        try {
            jobs.count();
            database = "UP";
        } catch (RuntimeException e) {
            database = "DOWN";
        }
        String mlStatus;
        String mlMessage;
        try {
            ml.health();
            mlStatus = "UP";
            mlMessage = "ML service is reachable.";
        } catch (MlServiceUnavailableException | MlServiceException e) {
            mlStatus = "DOWN";
            mlMessage = e.getMessage();
        }
        String overall = "UP".equals(database) && "UP".equals(mlStatus) ? "UP" : "DEGRADED";
        return new HealthDto(overall, database, mlStatus, mlMessage, Instant.now());
    }

    /** Real counts for the dashboard. */
    @GetMapping("/summary")
    public DashboardSummaryDto summary() {
        long active = jobs.countByStatus(JobStatus.QUEUED) + jobs.countByStatus(JobStatus.RUNNING);
        long available = models.countByAvailableTrue();
        return new DashboardSummaryDto(jobs.count(), jobs.countByStatus(JobStatus.COMPLETED),
                jobs.countByStatus(JobStatus.FAILED), active, available, predictions.count());
    }
}
