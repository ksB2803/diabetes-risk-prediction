package edu.ait.diabetesrisk.service;

import com.fasterxml.jackson.databind.JsonNode;
import edu.ait.diabetesrisk.client.MlServiceClient;
import edu.ait.diabetesrisk.client.MlServiceException;
import edu.ait.diabetesrisk.dto.PredictionHistoryDto;
import edu.ait.diabetesrisk.dto.PredictionRequest;
import edu.ait.diabetesrisk.dto.PredictionResponseDto;
import edu.ait.diabetesrisk.entity.PredictionHistory;
import edu.ait.diabetesrisk.entity.TrainedModel;
import edu.ait.diabetesrisk.exception.ApiException;
import edu.ait.diabetesrisk.repository.PredictionHistoryRepository;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
public class PredictionService {

    public static final String DISCLAIMER = "Educational project only. This model is not a medical device and cannot "
            + "diagnose diabetes. Results depend on the uploaded dataset and model quality. For health concerns, "
            + "consult a qualified healthcare professional.";

    private final ModelService modelService;
    private final MlServiceClient ml;
    private final PredictionHistoryRepository history;

    public PredictionService(ModelService modelService, MlServiceClient ml, PredictionHistoryRepository history) {
        this.modelService = modelService;
        this.ml = ml;
        this.history = history;
    }

    public PredictionResponseDto predict(Long modelId, PredictionRequest request) {
        TrainedModel model = modelService.find(modelId);
        if (!model.isAvailable()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "MODEL_UNAVAILABLE", "Model " + modelId + " is no longer available.");
        }
        validateAgainstSchema(modelService.featureSchema(model), request.values());

        JsonNode result;
        try {
            result = ml.predict(model.getMlModelId(), request.values());
        } catch (MlServiceException e) {
            if (e.getMlStatus() == 404) {
                modelService.markUnavailable(model);
                throw new ApiException(HttpStatus.NOT_FOUND, "MODEL_UNAVAILABLE",
                        "The saved file for model " + modelId + " is missing. Train a new model.");
            }
            throw e;
        }

        Double probability = result.path("probability_positive").isNumber()
                ? result.path("probability_positive").asDouble() : null;
        Map<String, Double> classProbabilities = new LinkedHashMap<>();
        for (Map.Entry<String, JsonNode> e : result.path("class_probabilities").properties()) {
            classProbabilities.put(e.getKey(), e.getValue().asDouble());
        }
        List<String> warnings = new ArrayList<>();
        result.path("warnings").forEach(w -> warnings.add(w.asText()));

        PredictionHistory record = new PredictionHistory();
        record.setModelId(model.getId());
        record.setAlgorithm(model.getAlgorithm());
        record.setPredictedClass(result.path("predicted_class").asText());
        record.setPositiveClass(result.path("is_positive_class").asBoolean());
        record.setProbabilityPositive(probability);
        record.setCreatedAt(Instant.now());
        PredictionHistory saved = history.save(record);

        return new PredictionResponseDto(
                saved.getId(),
                model.getId(),
                model.getDisplayName(),
                saved.getPredictedClass(),
                saved.isPositiveClass(),
                result.path("positive_class").asText(model.getPositiveClass()),
                result.path("negative_class").asText(model.getNegativeClass()),
                probability,
                classProbabilities.isEmpty() ? null : classProbabilities,
                warnings,
                DISCLAIMER,
                saved.getCreatedAt());
    }

    /** Checks names and basic types before calling the ML service, for clear 400 errors. */
    private static void validateAgainstSchema(JsonNode features, Map<String, Object> values) {
        Set<String> expected = new LinkedHashSet<>();
        Map<String, String> types = new LinkedHashMap<>();
        for (JsonNode f : features) {
            expected.add(f.path("name").asText());
            types.put(f.path("name").asText(), f.path("type").asText());
        }
        List<String> details = new ArrayList<>();
        List<String> missing = expected.stream().filter(n -> !values.containsKey(n)).toList();
        List<String> extra = values.keySet().stream().filter(n -> !expected.contains(n)).toList();
        if (!missing.isEmpty()) {
            details.add("Missing: " + String.join(", ", missing));
        }
        if (!extra.isEmpty()) {
            details.add("Not used by this model: " + String.join(", ", extra));
        }
        if (!details.isEmpty()) {
            throw ApiException.badRequest("INCOMPATIBLE_FEATURES",
                    "The input does not match the features this model was trained on.", details);
        }
        for (Map.Entry<String, Object> e : values.entrySet()) {
            Object v = e.getValue();
            if (v == null || (v instanceof String str && str.isBlank())) {
                details.add("\"" + e.getKey() + "\" needs a value.");
            } else if ("numeric".equals(types.get(e.getKey()))) {
                if (!(v instanceof Number) && !(v instanceof String text && isNumber(text))) {
                    details.add("\"" + e.getKey() + "\" must be a number.");
                }
            } else if (!(v instanceof String) && !(v instanceof Number) && !(v instanceof Boolean)) {
                details.add("\"" + e.getKey() + "\" must be a single value.");
            }
        }
        if (!details.isEmpty()) {
            throw ApiException.badRequest("INVALID_VALUE", "Some values are missing or have the wrong type.", details);
        }
    }

    private static boolean isNumber(String s) {
        try {
            double d = Double.parseDouble(s.trim());
            return Double.isFinite(d);
        } catch (NumberFormatException e) {
            return false;
        }
    }

    public List<PredictionHistoryDto> recent(int limit) {
        int size = Math.max(1, Math.min(limit, 200));
        return history.findAllByOrderByCreatedAtDesc(PageRequest.of(0, size)).stream()
                .map(PredictionHistoryDto::from).toList();
    }
}
