package edu.ait.diabetesrisk.client;

import java.util.List;

/** The ML service answered with an error status and a structured error body. */
public class MlServiceException extends RuntimeException {

    private final int mlStatus;
    private final String code;
    private final List<String> details;

    public MlServiceException(int mlStatus, String code, String message, List<String> details) {
        super(message);
        this.mlStatus = mlStatus;
        this.code = code;
        this.details = details == null ? List.of() : List.copyOf(details);
    }

    public int getMlStatus() { return mlStatus; }
    public String getCode() { return code; }
    public List<String> getDetails() { return details; }
}
