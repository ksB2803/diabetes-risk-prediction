package edu.ait.diabetesrisk.exception;

import edu.ait.diabetesrisk.client.MlServiceException;
import edu.ait.diabetesrisk.client.MlServiceUnavailableException;
import edu.ait.diabetesrisk.dto.ApiError;
import jakarta.validation.ConstraintViolationException;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.HandlerMethodValidationException;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.MultipartException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

/**
 * Maps every failure to a safe {@link ApiError}. Stack traces and file paths are
 * logged on the server only and never sent to the browser.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    private static ResponseEntity<ApiError> error(HttpStatus status, String code, String message, List<String> details) {
        return ResponseEntity.status(status).body(ApiError.of(status.value(), code, message, details));
    }

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiError> handleApi(ApiException e) {
        return error(e.getStatus(), e.getCode(), e.getMessage(), e.getDetails());
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiError> handleInvalid(MethodArgumentNotValidException e) {
        List<String> details = e.getBindingResult().getFieldErrors().stream()
                .map(f -> f.getField() + ": " + f.getDefaultMessage())
                .toList();
        return error(HttpStatus.BAD_REQUEST, "VALIDATION_ERROR", "Some settings are missing or invalid.", details);
    }

    @ExceptionHandler(HandlerMethodValidationException.class)
    public ResponseEntity<ApiError> handleMethodValidation(HandlerMethodValidationException e) {
        List<String> details = e.getAllErrors().stream().map(err -> String.valueOf(err.getDefaultMessage())).toList();
        return error(HttpStatus.BAD_REQUEST, "VALIDATION_ERROR", "Some settings are missing or invalid.", details);
    }

    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ApiError> handleConstraint(ConstraintViolationException e) {
        List<String> details = e.getConstraintViolations().stream()
                .map(v -> v.getPropertyPath() + ": " + v.getMessage())
                .toList();
        return error(HttpStatus.BAD_REQUEST, "VALIDATION_ERROR", "Some settings are missing or invalid.", details);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ApiError> handleUnreadable(HttpMessageNotReadableException e) {
        return error(HttpStatus.BAD_REQUEST, "MALFORMED_REQUEST", "The request body is not valid JSON.", List.of());
    }

    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ApiError> handleMissingPart(MissingServletRequestPartException e) {
        return error(HttpStatus.BAD_REQUEST, "MISSING_PART",
                "The request is missing the '" + e.getRequestPartName() + "' part.", List.of());
    }

    @ExceptionHandler({MissingServletRequestParameterException.class, MethodArgumentTypeMismatchException.class})
    public ResponseEntity<ApiError> handleBadParam(Exception e) {
        return error(HttpStatus.BAD_REQUEST, "BAD_PARAMETER", "A request parameter is missing or has the wrong type.", List.of());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiError> handleTooLarge(MaxUploadSizeExceededException e) {
        return error(HttpStatus.PAYLOAD_TOO_LARGE, "FILE_TOO_LARGE", "The file is larger than the upload limit.", List.of());
    }

    @ExceptionHandler(MultipartException.class)
    public ResponseEntity<ApiError> handleMultipart(MultipartException e) {
        return error(HttpStatus.BAD_REQUEST, "BAD_UPLOAD", "The upload could not be read. Send the file as multipart/form-data.", List.of());
    }

    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    public ResponseEntity<ApiError> handleMediaType(HttpMediaTypeNotSupportedException e) {
        return error(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "UNSUPPORTED_MEDIA_TYPE", "This content type is not supported here.", List.of());
    }

    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResponseEntity<ApiError> handleMethod(HttpRequestMethodNotSupportedException e) {
        return error(HttpStatus.METHOD_NOT_ALLOWED, "METHOD_NOT_ALLOWED", "This HTTP method is not allowed here.", List.of());
    }

    @ExceptionHandler(NoResourceFoundException.class)
    public ResponseEntity<ApiError> handleNoResource(NoResourceFoundException e) {
        return error(HttpStatus.NOT_FOUND, "NOT_FOUND", "Nothing was found at this address.", List.of());
    }

    @ExceptionHandler(MlServiceUnavailableException.class)
    public ResponseEntity<ApiError> handleMlDown(MlServiceUnavailableException e) {
        return error(HttpStatus.SERVICE_UNAVAILABLE, "ML_SERVICE_UNAVAILABLE", e.getMessage(), List.of());
    }

    @ExceptionHandler(MlServiceException.class)
    public ResponseEntity<ApiError> handleMl(MlServiceException e) {
        int s = e.getMlStatus();
        if (s == 413) {
            return error(HttpStatus.PAYLOAD_TOO_LARGE, e.getCode(), e.getMessage(), e.getDetails());
        }
        if (s == 404) {
            return error(HttpStatus.NOT_FOUND, e.getCode(), e.getMessage(), e.getDetails());
        }
        if (s == 429) {
            return error(HttpStatus.TOO_MANY_REQUESTS, e.getCode(), e.getMessage(), e.getDetails());
        }
        if (s == 401 || s == 403) {
            log.error("ML service rejected the internal API key; check ML_INTERNAL_API_KEY on both services");
            return error(HttpStatus.BAD_GATEWAY, "ML_AUTH_FAILED", "The backend is not authorised to use the ML service.", List.of());
        }
        if (s >= 400 && s < 500) {
            return error(HttpStatus.BAD_REQUEST, e.getCode(), e.getMessage(), e.getDetails());
        }
        return error(HttpStatus.BAD_GATEWAY, "ML_SERVICE_ERROR", "The ML service failed while processing the request.", List.of());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleUnexpected(Exception e) {
        log.error("Unexpected error", e);
        return error(HttpStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "Something went wrong on the server. Try again.", List.of());
    }
}
