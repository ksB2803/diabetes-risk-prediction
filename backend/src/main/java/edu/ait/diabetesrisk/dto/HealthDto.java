package edu.ait.diabetesrisk.dto;

import java.time.Instant;

/** Overall status plus the status of each dependency. */
public record HealthDto(String status, String database, String mlService, String mlServiceMessage, Instant checkedAt) {
}
