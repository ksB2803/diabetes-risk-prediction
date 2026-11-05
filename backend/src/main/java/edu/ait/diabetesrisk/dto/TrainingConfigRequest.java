package edu.ait.diabetesrisk.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.List;

/** Answers from the training wizard, sent as the JSON `config` part of the upload. */
public record TrainingConfigRequest(
        @NotBlank @Size(max = 200) String targetColumn,
        @NotEmpty @Size(max = 100) List<@NotBlank @Size(max = 200) String> featureColumns,
        @NotNull @DecimalMin("0.5") @DecimalMax("0.9") Double trainFraction,
        @NotEmpty @Size(max = 3)
        List<@NotNull @Pattern(regexp = "logistic_regression|random_forest|decision_tree",
                message = "must be logistic_regression, random_forest or decision_tree") String> algorithms,
        @NotNull @Pattern(regexp = "roc_auc|f1|recall|accuracy|precision",
                message = "must be roc_auc, f1, recall, accuracy or precision") String scoringMetric,
        @NotNull @Pattern(regexp = "quick|balanced|thorough", message = "must be quick, balanced or thorough")
        String tuningLevel,
        @NotNull @Min(0) @Max(2147483647) Integer randomSeed,
        @Size(max = 200) String positiveClass,
        @Pattern(regexp = "real|synthetic|unknown", message = "must be real, synthetic or unknown") String datasetOrigin) {
}
