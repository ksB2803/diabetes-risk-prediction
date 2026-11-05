package edu.ait.diabetesrisk.client;

/** The ML service could not be reached or did not answer in time. */
public class MlServiceUnavailableException extends RuntimeException {

    public MlServiceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
