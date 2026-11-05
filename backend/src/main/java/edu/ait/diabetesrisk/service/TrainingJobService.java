package edu.ait.diabetesrisk.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import edu.ait.diabetesrisk.client.MlServiceClient;
import edu.ait.diabetesrisk.dto.TrainingConfigRequest;
import edu.ait.diabetesrisk.dto.TrainingJobDto;
import edu.ait.diabetesrisk.entity.JobStatus;
import edu.ait.diabetesrisk.entity.TrainedModel;
import edu.ait.diabetesrisk.entity.TrainingJob;
import edu.ait.diabetesrisk.exception.ApiException;
import edu.ait.diabetesrisk.repository.TrainedModelRepository;
import edu.ait.diabetesrisk.repository.TrainingJobRepository;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/** Creates training jobs, queues them on the bounded worker pool and serves their results. */
@Service
public class TrainingJobService {

    private final TrainingJobRepository jobs;
    private final TrainedModelRepository models;
    private final UploadValidator uploadValidator;
    private final StorageService storage;
    private final MlServiceClient ml;
    private final TrainingWorker worker;
    private final ThreadPoolExecutor executor;
    private final ObjectMapper mapper;

    public TrainingJobService(TrainingJobRepository jobs, TrainedModelRepository models, UploadValidator uploadValidator,
                              StorageService storage, MlServiceClient ml, TrainingWorker worker,
                              @Qualifier("trainingExecutor") ThreadPoolExecutor executor, ObjectMapper mapper) {
        this.jobs = jobs;
        this.models = models;
        this.uploadValidator = uploadValidator;
        this.storage = storage;
        this.ml = ml;
        this.worker = worker;
        this.executor = executor;
        this.mapper = mapper;
    }

    /** Profiles a CSV without storing it (wizard step 1). */
    public JsonNode inspect(MultipartFile file) {
        byte[] bytes = uploadValidator.readCsv(file);
        return ml.inspect(bytes, UploadValidator.displayName(file.getOriginalFilename()));
    }

    public TrainingJobDto create(MultipartFile file, TrainingConfigRequest config) {
        validateConfig(config);
        byte[] bytes = uploadValidator.readCsv(file);
        if (executor.getQueue().remainingCapacity() == 0) {
            throw ApiException.tooManyRequests("Too many training jobs are waiting. Try again in a few minutes.");
        }

        String stored = storage.storeUpload(bytes);
        TrainingJob job = new TrainingJob();
        job.setOriginalFilename(UploadValidator.displayName(file.getOriginalFilename()));
        job.setStoredFilename(stored);
        job.setFileSizeBytes(bytes.length);
        job.setDatasetOrigin(config.datasetOrigin() == null ? "unknown" : config.datasetOrigin());
        job.setTargetColumn(config.targetColumn().trim());
        job.setFeatureColumns(new ArrayList<>(config.featureColumns()));
        job.setTrainFraction(config.trainFraction());
        job.setAlgorithms(new ArrayList<>(config.algorithms()));
        job.setScoringMetric(config.scoringMetric());
        job.setTuningLevel(config.tuningLevel());
        job.setRandomSeed(config.randomSeed());
        job.setPositiveClass(config.positiveClass() == null || config.positiveClass().isBlank()
                ? null : config.positiveClass().trim());
        job.setStatus(JobStatus.QUEUED);
        job.setStage(TrainingWorker.STAGE_QUEUED);
        job.setCreatedAt(Instant.now());
        TrainingJob saved = jobs.save(job);

        try {
            Long id = saved.getId();
            executor.execute(() -> worker.run(id));
        } catch (RejectedExecutionException e) {
            jobs.delete(saved);
            storage.deleteQuietly(stored);
            throw ApiException.tooManyRequests("Too many training jobs are waiting. Try again in a few minutes.");
        }
        return TrainingJobDto.from(saved);
    }

    private void validateConfig(TrainingConfigRequest config) {
        List<String> problems = new ArrayList<>();
        String target = config.targetColumn().trim();
        if (config.featureColumns().stream().anyMatch(f -> f.trim().equals(target))) {
            problems.add("The target column cannot also be a feature.");
        }
        if (new HashSet<>(config.featureColumns()).size() != config.featureColumns().size()) {
            problems.add("Each feature can only be selected once.");
        }
        if (new HashSet<>(config.algorithms()).size() != config.algorithms().size()) {
            problems.add("Each algorithm can only be selected once.");
        }
        if (!problems.isEmpty()) {
            throw ApiException.badRequest("INVALID_CONFIG", "The training settings are not valid.", problems);
        }
    }

    public List<TrainingJobDto> list(int limit) {
        int size = Math.max(1, Math.min(limit, 200));
        return jobs.findAllByOrderByCreatedAtDesc(PageRequest.of(0, size)).stream().map(TrainingJobDto::from).toList();
    }

    public TrainingJobDto get(Long id) {
        return TrainingJobDto.from(find(id));
    }

    /** Results document for a completed job, with internal IDs and file names removed. */
    public JsonNode results(Long id) {
        TrainingJob job = find(id);
        if (job.getStatus() != JobStatus.COMPLETED || job.getResultsJson() == null) {
            throw ApiException.conflict("RESULTS_NOT_READY",
                    "Results are not available for job " + id + " (status: " + job.getStatus().name() + ").");
        }
        ObjectNode root;
        try {
            root = (ObjectNode) mapper.readTree(job.getResultsJson());
        } catch (JsonProcessingException | ClassCastException e) {
            throw new IllegalStateException("Stored results are not valid JSON", e);
        }
        Map<String, TrainedModel> byMlId = new HashMap<>();
        for (TrainedModel m : models.findByTrainingJobIdOrderByIdAsc(id)) {
            byMlId.put(m.getMlModelId(), m);
        }
        for (JsonNode node : root.path("models")) {
            ObjectNode model = (ObjectNode) node;
            TrainedModel saved = byMlId.get(model.path("model_id").asText());
            model.remove("model_id");
            if (saved != null) {
                model.put("id", saved.getId());
                model.put("selected", saved.isSelectedByCv());
                model.put("available", saved.isAvailable());
            }
            if (model.get("artifact") instanceof ObjectNode artifact) {
                artifact.remove("file");
            }
        }
        root.remove("selected_model_id");
        root.remove("job_id");
        if (job.getSelectedModelId() != null) {
            root.put("selected_model_id", job.getSelectedModelId());
        }
        root.set("job", mapper.valueToTree(TrainingJobDto.from(job)));
        return root;
    }

    /** Flat CSV report: one row per model, CV score next to held-out test metrics. */
    public String csvReport(Long id) {
        JsonNode results = results(id);
        StringBuilder sb = new StringBuilder();
        sb.append("job_id,model_id,algorithm,selected_by_cv,scoring_metric,cv_score,cv_std,")
          .append("test_accuracy,test_precision,test_recall,test_f1,test_roc_auc,train_rows,test_rows,random_seed,best_params\n");
        String metric = results.path("scoring_metric").asText();
        JsonNode split = results.path("split");
        for (JsonNode m : results.path("models")) {
            JsonNode t = m.path("test_metrics");
            List<String> row = List.of(
                    String.valueOf(id),
                    m.path("id").asText(),
                    m.path("display_name").asText(),
                    String.valueOf(m.path("selected").asBoolean()),
                    metric,
                    num(m.path("cv_score")),
                    num(m.path("cv_std")),
                    num(t.path("accuracy")),
                    num(t.path("precision")),
                    num(t.path("recall")),
                    num(t.path("f1")),
                    num(t.path("roc_auc")),
                    split.path("train_rows").asText(),
                    split.path("test_rows").asText(),
                    split.path("random_seed").asText(),
                    m.path("best_params").toString());
            sb.append(String.join(",", row.stream().map(TrainingJobService::csvCell).toList())).append('\n');
        }
        return sb.toString();
    }

    private static String num(JsonNode n) {
        return n.isNumber() ? n.asText() : "";
    }

    /** Quotes a CSV cell and neutralises spreadsheet formula injection. */
    static String csvCell(String value) {
        String v = value == null ? "" : value;
        if (!v.isEmpty() && "=+-@".indexOf(v.charAt(0)) >= 0 && !v.matches("-?\\d+(\\.\\d+)?([eE]-?\\d+)?")) {
            v = "'" + v;
        }
        return "\"" + v.replace("\"", "\"\"") + "\"";
    }

    private TrainingJob find(Long id) {
        return jobs.findById(id).orElseThrow(() -> ApiException.notFound("Training job", id));
    }
}
