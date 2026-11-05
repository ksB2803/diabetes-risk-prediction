package edu.ait.diabetesrisk.exception;

import java.util.List;
import org.springframework.http.HttpStatus;

/**
 * Base class for expected, user-facing errors. The message must be safe to show
 * to a browser: no file paths, stack traces or raw data.
 */
public class ApiException extends RuntimeException {

    private final HttpStatus status;
    private final String code;
    private final List<String> details;

    public ApiException(HttpStatus status, String code, String message) {
        this(status, code, message, List.of());
    }

    public ApiException(HttpStatus status, String code, String message, List<String> details) {
        super(message);
        this.status = status;
        this.code = code;
        this.details = details == null ? List.of() : List.copyOf(details);
    }

    public HttpStatus getStatus() { return status; }
    public String getCode() { return code; }
    public List<String> getDetails() { return details; }

    public static ApiException badRequest(String code, String message) {
        return new ApiException(HttpStatus.BAD_REQUEST, code, message);
    }

    public static ApiException badRequest(String code, String message, List<String> details) {
        return new ApiException(HttpStatus.BAD_REQUEST, code, message, details);
    }

    public static ApiException notFound(String what, Object id) {
        return new ApiException(HttpStatus.NOT_FOUND, "NOT_FOUND", what + " " + id + " was not found.");
    }

    public static ApiException conflict(String code, String message) {
        return new ApiException(HttpStatus.CONFLICT, code, message);
    }

    public static ApiException tooManyRequests(String message) {
        return new ApiException(HttpStatus.TOO_MANY_REQUESTS, "TOO_MANY_JOBS", message);
    }
}
