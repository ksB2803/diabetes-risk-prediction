package edu.ait.diabetesrisk.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.NullNode;
import edu.ait.diabetesrisk.dto.TrainedModelDto;
import edu.ait.diabetesrisk.entity.TrainedModel;
import edu.ait.diabetesrisk.entity.TrainingJob;
import edu.ait.diabetesrisk.exception.ApiException;
import edu.ait.diabetesrisk.repository.TrainedModelRepository;
import edu.ait.diabetesrisk.repository.TrainingJobRepository;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

@Service
public class ModelService {

    private final TrainedModelRepository models;
    private final TrainingJobRepository jobs;
    private final ObjectMapper mapper;

    public ModelService(TrainedModelRepository models, TrainingJobRepository jobs, ObjectMapper mapper) {
        this.models = models;
        this.jobs = jobs;
        this.mapper = mapper;
    }

    public List<TrainedModelDto> list(boolean includeUnavailable) {
        List<TrainedModel> all = models.findAllByOrderByCreatedAtDesc();
        Map<Long, TrainingJob> jobById = jobs.findAllById(all.stream().map(TrainedModel::getTrainingJobId).distinct().toList())
                .stream().collect(Collectors.toMap(TrainingJob::getId, Function.identity()));
        return all.stream()
                .filter(m -> includeUnavailable || m.isAvailable())
                .map(m -> toDto(m, jobById.get(m.getTrainingJobId())))
                .toList();
    }

    public TrainedModelDto get(Long id) {
        TrainedModel model = find(id);
        return toDto(model, jobs.findById(model.getTrainingJobId()).orElse(null));
    }

    public TrainedModel find(Long id) {
        return models.findById(id).orElseThrow(() -> ApiException.notFound("Model", id));
    }

    public void markUnavailable(TrainedModel model) {
        model.setAvailable(false);
        models.save(model);
    }

    public JsonNode featureSchema(TrainedModel model) {
        return parse(model.getFeatureSchemaJson());
    }

    private TrainedModelDto toDto(TrainedModel m, TrainingJob job) {
        return new TrainedModelDto(
                m.getId(),
                m.getTrainingJobId(),
                m.getAlgorithm(),
                m.getDisplayName(),
                m.getVersion(),
                m.getTargetColumn(),
                m.getPositiveClass(),
                m.getNegativeClass(),
                m.getSelectionMetric(),
                m.getCvScore(),
                m.getCvStd(),
                parse(m.getTestMetricsJson()),
                parse(m.getBestParamsJson()),
                parse(m.getFeatureSchemaJson()),
                m.isSelectedByCv(),
                m.isAvailable(),
                job == null ? "unknown" : job.getDatasetOrigin(),
                job == null ? null : job.getOriginalFilename(),
                m.getCreatedAt());
    }

    private JsonNode parse(String json) {
        if (json == null || json.isBlank()) {
            return NullNode.getInstance();
        }
        try {
            return mapper.readTree(json);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Stored model metadata is not valid JSON", e);
        }
    }
}
