package edu.ait.diabetesrisk.client;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.ait.diabetesrisk.config.AppProperties;
import java.net.http.HttpClient;
import java.net.http.HttpTimeoutException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

/**
 * The only component that talks to the internal Python ML service.
 * Transport: multipart upload of the CSV bytes plus a JSON config string.
 * Errors are mapped to {@link MlServiceException} (structured ML error) or
 * {@link MlServiceUnavailableException} (connection refused / timeout).
 */
@Component
public class MlServiceClient {

    private static final Logger log = LoggerFactory.getLogger(MlServiceClient.class);
    private static final String API_KEY_HEADER = "X-Internal-Api-Key";

    private final RestClient quickClient;
    private final RestClient trainingClient;
    private final ObjectMapper mapper;

    public MlServiceClient(AppProperties props, ObjectMapper mapper) {
        this.mapper = mapper;
        AppProperties.Ml ml = props.getMl();
        this.quickClient = build(ml.getBaseUrl(), ml.getConnectTimeout(), ml.getRequestTimeout(), ml.getApiKey());
        this.trainingClient = build(ml.getBaseUrl(), ml.getConnectTimeout(), ml.getTrainingTimeout(), ml.getApiKey());
    }

    private static RestClient build(String baseUrl, Duration connect, Duration read, String apiKey) {
        HttpClient http = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(connect)
                .build();
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(http);
        factory.setReadTimeout(read);
        RestClient.Builder builder = RestClient.builder().baseUrl(baseUrl).requestFactory(factory);
        if (apiKey != null && !apiKey.isBlank()) {
            builder.defaultHeader(API_KEY_HEADER, apiKey);
        }
        return builder.build();
    }

    /** GET /ml/health. Returns the health body or throws MlServiceUnavailableException. */
    public JsonNode health() {
        return call("health check", () -> quickClient.get().uri("/ml/health").retrieve().body(JsonNode.class));
    }

    /** POST /ml/inspect with the uploaded bytes. */
    public JsonNode inspect(byte[] csv, String filename) {
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", namedBytes(csv, filename));
        return call("dataset inspection", () -> quickClient.post()
                .uri("/ml/inspect")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(JsonNode.class));
    }

    /** POST /ml/train with a stored CSV (controlled storage path, never browser-supplied). */
    public JsonNode train(Path storedCsv, Map<String, Object> config) {
        String configJson;
        try {
            configJson = mapper.writeValueAsString(config);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Could not serialise training config", e);
        }
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", new FileSystemResource(storedCsv));
        parts.add("config", configJson);
        return call("training", () -> trainingClient.post()
                .uri("/ml/train")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(JsonNode.class));
    }

    /** POST /ml/predict. */
    public JsonNode predict(String mlModelId, Map<String, Object> values) {
        Map<String, Object> body = Map.of("model_id", mlModelId, "values", values);
        return call("prediction", () -> quickClient.post()
                .uri("/ml/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(JsonNode.class));
    }

    private static ByteArrayResource namedBytes(byte[] bytes, String filename) {
        return new ByteArrayResource(bytes) {
            @Override
            public String getFilename() {
                return filename;
            }
        };
    }

    private JsonNode call(String operation, Supplier<JsonNode> request) {
        try {
            JsonNode body = request.get();
            if (body == null) {
                throw new MlServiceException(502, "ML_EMPTY_RESPONSE", "The ML service returned an empty response.", List.of());
            }
            return body;
        } catch (RestClientResponseException e) {
            throw toMlException(e);
        } catch (ResourceAccessException e) {
            boolean timeout = hasCause(e, HttpTimeoutException.class);
            log.warn("ML service {} failed: {}", operation, timeout ? "timeout" : "unreachable");
            throw new MlServiceUnavailableException(timeout
                    ? "The ML service did not respond in time."
                    : "The ML service is not reachable. Make sure it is running.", e);
        } catch (RestClientException e) {
            log.warn("ML service {} failed: {}", operation, e.getClass().getSimpleName());
            throw new MlServiceUnavailableException("The ML service could not be contacted.", e);
        }
    }

    private MlServiceException toMlException(RestClientResponseException e) {
        int status = e.getStatusCode().value();
        String code = "ML_ERROR";
        String message = "The ML service rejected the request.";
        List<String> details = new ArrayList<>();
        try {
            JsonNode error = mapper.readTree(e.getResponseBodyAsString()).path("error");
            if (error.hasNonNull("code")) {
                code = error.get("code").asText();
            }
            if (error.hasNonNull("message")) {
                message = error.get("message").asText();
            }
            error.path("details").forEach(d -> details.add(d.asText()));
        } catch (JsonProcessingException | RuntimeException ignored) {
            // Non-JSON body: keep the generic message rather than echoing it.
        }
        if (status >= 500) {
            message = "The ML service failed while processing the request.";
            details.clear();
        }
        return new MlServiceException(status, code, message, details);
    }

    private static boolean hasCause(Throwable t, Class<? extends Throwable> type) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (type.isInstance(c)) {
                return true;
            }
        }
        return false;
    }
}
