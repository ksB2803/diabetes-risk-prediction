package edu.ait.diabetesrisk.dto;

import com.fasterxml.jackson.databind.JsonNode;
import java.time.Instant;

/**
 * Public view of a saved model. Artifact IDs and file names stay internal.
 * {@code features} is the input schema used to build the prediction form.
 */
public record TrainedModelDto(
        Long id,
        Long trainingJobId,
        String algorithm,
        String displayName,
        String version,
        String targetColumn,
        String positiveClass,
        String negativeClass,
        String selectionMetric,
        Double cvScore,
        Double cvStd,
        JsonNode testMetrics,
        JsonNode bestParams,
        JsonNode features,
        boolean selectedByCv,
        boolean available,
        String datasetOrigin,
        String datasetFilename,
        Instant createdAt) {
}
