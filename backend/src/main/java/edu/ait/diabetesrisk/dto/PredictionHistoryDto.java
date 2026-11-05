package edu.ait.diabetesrisk.dto;

import edu.ait.diabetesrisk.entity.PredictionHistory;
import java.time.Instant;

public record PredictionHistoryDto(
        Long id,
        Long modelId,
        String algorithm,
        String predictedClass,
        boolean positiveClassPredicted,
        Double probabilityPositive,
        Instant createdAt) {

    public static PredictionHistoryDto from(PredictionHistory p) {
        return new PredictionHistoryDto(p.getId(), p.getModelId(), p.getAlgorithm(), p.getPredictedClass(),
                p.isPositiveClass(), p.getProbabilityPositive(), p.getCreatedAt());
    }
}
