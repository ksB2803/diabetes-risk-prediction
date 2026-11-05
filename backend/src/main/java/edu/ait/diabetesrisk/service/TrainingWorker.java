package edu.ait.diabetesrisk.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.ait.diabetesrisk.client.MlServiceClient;
import edu.ait.diabetesrisk.client.MlServiceException;
import edu.ait.diabetesrisk.client.MlServiceUnavailableException;
import edu.ait.diabetesrisk.config.AppProperties;
import edu.ait.diabetesrisk.entity.JobStatus;
import edu.ait.diabetesrisk.entity.TrainedModel;
import edu.ait.diabetesrisk.entity.TrainingJob;
import edu.ait.diabetesrisk.repository.TrainedModelRepository;
import edu.ait.diabetesrisk.repository.TrainingJobRepository;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Runs one training job on a worker thread: sends the stored CSV and config to
 * the ML service, then records the results and saved models in H2.
 */
@Component
public class TrainingWorker {

    private static final Logger log = LoggerFactory.getLogger(TrainingWorker.class);

    static final String STAGE_QUEUED = "Waiting for a free training slot";
    static final String STAGE_TRAINING = "Validating data, tuning and evaluating models";
    static final String STAGE_SAVING = "Saving results";
    static final String STAGE_DONE = "Finished";
    static final String STAGE_FAILED = "Failed";

    private final TrainingJobRepository jobs;
    private final TrainedModelRepository models;
    private final MlServiceClient ml;
    private final StorageService storage;
    private final AppProperties props;
    private final ObjectMapper mapper;
    private final TransactionTemplate tx;

    public TrainingWorker(TrainingJobRepository jobs, TrainedModelRepository models, MlServiceClient ml,
                          StorageService storage, AppProperties props, ObjectMapper mapper,
                          PlatformTransactionManager txManager) {
        this.jobs = jobs;
        this.models = models;
        this.ml = ml;
        this.storage = storage;
        this.props = props;
        this.mapper = mapper;
        this.tx = new TransactionTemplate(txManager);
    }

    public void run(Long jobId) {
        TrainingJob job = jobs.findById(jobId).orElse(null);
        if (job == null || job.getStatus() != JobStatus.QUEUED) {
            return;
        }
        job.setStatus(JobStatus.RUNNING);
        job.setStage(STAGE_TRAINING);
        job.setStartedAt(Instant.now());
        jobs.save(job);

        try {
            JsonNode result = ml.train(storage.resolve(job.getStoredFilename()), mlConfig(job));
            job.setStage(STAGE_SAVING);
            jobs.save(job);
            tx.executeWithoutResult(status -> complete(jobId, result));
            log.info("Training job {} completed", jobId);
        } catch (MlServiceException e) {
            fail(jobId, e.getMessage() + (e.getDetails().isEmpty() ? "" : " " + String.join("; ", e.getDetails())));
            log.info("Training job {} rejected by ML service: {}", jobId, e.getCode());
        } catch (MlServiceUnavailableException e) {
            fail(jobId, e.getMessage() + " Start the ML service and try again.");
        } catch (RuntimeException e) {
            log.error("Training job {} failed unexpectedly", jobId, e);
            fail(jobId, "Training failed because of an unexpected server error.");
        } finally {
            if (!props.getStorage().isKeepUploads()) {
                storage.deleteQuietly(job.getStoredFilename());
            }
        }
    }

    private Map<String, Object> mlConfig(TrainingJob job) {
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("job_id", String.valueOf(job.getId()));
        config.put("target_column", job.getTargetColumn());
        config.put("feature_columns", job.getFeatureColumns());
        config.put("train_fraction", job.getTrainFraction());
        config.put("algorithms", job.getAlgorithms());
        config.put("scoring_metric", job.getScoringMetric());
        config.put("tuning_level", job.getTuningLevel());
        config.put("random_seed", job.getRandomSeed());
        config.put("task_type", "binary_classification");
        if (job.getPositiveClass() != null && !job.getPositiveClass().isBlank()) {
            config.put("positive_class", job.getPositiveClass());
        }
        return config;
    }

    private void complete(Long jobId, JsonNode result) {
        TrainingJob job = jobs.findById(jobId).orElseThrow();
        String selectedMlId = result.path("selected_model_id").asText();
        JsonNode schema = result.path("schema");
        Long selectedId = null;
        Double bestCv = null;

        for (JsonNode m : result.path("models")) {
            TrainedModel model = new TrainedModel();
            model.setTrainingJobId(jobId);
            model.setMlModelId(m.path("model_id").asText());
            model.setArtifactFile(m.path("artifact").path("file").asText());
            model.setAlgorithm(m.path("algorithm").asText());
            model.setDisplayName(m.path("display_name").asText());
            model.setVersion("pipeline v" + m.path("artifact").path("pipeline_version").asText("1")
                    + ", scikit-learn " + m.path("artifact").path("sklearn_version").asText("?"));
            model.setTargetColumn(job.getTargetColumn());
            model.setPositiveClass(schema.path("positive_class").asText(null));
            model.setNegativeClass(schema.path("negative_class").asText(null));
            model.setFeatureSchemaJson(write(schema.path("features")));
            model.setSelectionMetric(job.getScoringMetric());
            model.setCvScore(m.path("cv_score").isNumber() ? m.path("cv_score").asDouble() : null);
            model.setCvStd(m.path("cv_std").isNumber() ? m.path("cv_std").asDouble() : null);
            model.setTestMetricsJson(write(m.path("test_metrics")));
            model.setBestParamsJson(write(m.path("best_params")));
            model.setSelectedByCv(model.getMlModelId().equals(selectedMlId));
            model.setAvailable(true);
            model.setCreatedAt(Instant.now());
            TrainedModel saved = models.save(model);
            if (saved.isSelectedByCv()) {
                selectedId = saved.getId();
                bestCv = saved.getCvScore();
            }
        }

        JsonNode split = result.path("split");
        JsonNode summary = result.path("dataset_summary");
        job.setTotalRows(summary.path("n_rows_raw").isNumber() ? summary.path("n_rows_raw").asInt() : null);
        job.setRowsUsed(summary.path("n_rows_used").isNumber() ? summary.path("n_rows_used").asInt() : null);
        job.setTrainRows(split.path("train_rows").isNumber() ? split.path("train_rows").asInt() : null);
        job.setTestRows(split.path("test_rows").isNumber() ? split.path("test_rows").asInt() : null);
        job.setPositiveClass(result.path("class_mapping").path("positive_class").asText(job.getPositiveClass()));
        job.setSelectedModelId(selectedId);
        job.setBestCvScore(bestCv);
        job.setResultsJson(write(result));
        job.setStatus(JobStatus.COMPLETED);
        job.setStage(STAGE_DONE);
        job.setErrorMessage(null);
        job.setFinishedAt(Instant.now());
        jobs.save(job);
    }

    private void fail(Long jobId, String message) {
        jobs.findById(jobId).ifPresent(job -> {
            job.setStatus(JobStatus.FAILED);
            job.setStage(STAGE_FAILED);
            job.setErrorMessage(message.length() > 1000 ? message.substring(0, 997) + "..." : message);
            job.setFinishedAt(Instant.now());
            jobs.save(job);
        });
    }

    private String write(JsonNode node) {
        try {
            return mapper.writeValueAsString(node);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Could not serialise ML results", e);
        }
    }
}
