package edu.ait.diabetesrisk.dto;

import edu.ait.diabetesrisk.entity.TrainingJob;
import java.time.Instant;
import java.util.List;

/** Public view of a training job. The stored file location is intentionally absent. */
public record TrainingJobDto(
        Long id,
        String filename,
        long fileSizeBytes,
        String datasetOrigin,
        String targetColumn,
        List<String> featureColumns,
        double trainFraction,
        List<String> algorithms,
        String scoringMetric,
        String tuningLevel,
        int randomSeed,
        String positiveClass,
        String status,
        String stage,
        String errorMessage,
        Instant createdAt,
        Instant startedAt,
        Instant finishedAt,
        Integer totalRows,
        Integer rowsUsed,
        Integer trainRows,
        Integer testRows,
        Long selectedModelId,
        Double bestCvScore) {

    public static TrainingJobDto from(TrainingJob job) {
        return new TrainingJobDto(
                job.getId(),
                job.getOriginalFilename(),
                job.getFileSizeBytes(),
                job.getDatasetOrigin(),
                job.getTargetColumn(),
                List.copyOf(job.getFeatureColumns()),
                job.getTrainFraction(),
                List.copyOf(job.getAlgorithms()),
                job.getScoringMetric(),
                job.getTuningLevel(),
                job.getRandomSeed(),
                job.getPositiveClass(),
                job.getStatus().name(),
                job.getStage(),
                job.getErrorMessage(),
                job.getCreatedAt(),
                job.getStartedAt(),
                job.getFinishedAt(),
                job.getTotalRows(),
                job.getRowsUsed(),
                job.getTrainRows(),
                job.getTestRows(),
                job.getSelectedModelId(),
                job.getBestCvScore());
    }
}
