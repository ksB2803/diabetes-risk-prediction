package edu.ait.diabetesrisk.dto;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Result of a single prediction. {@code probabilityPositive} is a model-estimated
 * probability, not a calibrated personal medical risk.
 */
public record PredictionResponseDto(
        Long predictionId,
        Long modelId,
        String algorithm,
        String predictedClass,
        boolean positiveClassPredicted,
        String positiveClass,
        String negativeClass,
        Double probabilityPositive,
        Map<String, Double> classProbabilities,
        List<String> warnings,
        String disclaimer,
        Instant createdAt) {
}
