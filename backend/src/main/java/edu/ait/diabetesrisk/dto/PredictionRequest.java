package edu.ait.diabetesrisk.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.util.Map;

/** Feature values keyed by the exact feature names in the model schema. */
public record PredictionRequest(@NotNull @Size(min = 1, max = 200) Map<String, Object> values) {
}
