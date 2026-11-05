package edu.ait.diabetesrisk.dto;

import java.time.Instant;
import java.util.List;

/** Uniform error body for every public endpoint. */
public record ApiError(int status, String code, String message, List<String> details, Instant timestamp) {

    public static ApiError of(int status, String code, String message, List<String> details) {
        return new ApiError(status, code, message, details == null ? List.of() : details, Instant.now());
    }
}
