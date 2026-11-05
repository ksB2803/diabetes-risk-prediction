package edu.ait.diabetesrisk.dto;

/** Real counts from the database for the dashboard. */
public record DashboardSummaryDto(long totalJobs, long completedJobs, long failedJobs, long activeJobs,
                                  long availableModels, long predictions) {
}
