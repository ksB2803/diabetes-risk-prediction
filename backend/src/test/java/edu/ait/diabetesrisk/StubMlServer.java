package edu.ait.diabetesrisk;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Minimal stand-in for the Python ML service. It replays JSON responses that
 * were captured from the real service (src/test/resources/stub/*.json), so the
 * backend is tested against the actual response contract.
 */
final class StubMlServer {

    private final HttpServer server;
    private final String inspect;
    private final String trainTemplate;
    private final List<String> templateModelIds = new ArrayList<>();
    private final String predict;
    volatile String lastTrainRequest = "";

    StubMlServer() {
        try {
            inspect = resource("/stub/inspect-response.json");
            trainTemplate = resource("/stub/train-response.json");
            predict = resource("/stub/predict-response.json");
            JsonNode train = new ObjectMapper().readTree(trainTemplate);
            train.path("models").forEach(m -> templateModelIds.add(m.path("model_id").asText()));

            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/ml/health", ex -> send(ex, 200, "{\"status\":\"ok\",\"sklearn_version\":\"stub\",\"model_store_writable\":true}"));
            server.createContext("/ml/inspect", ex -> {
                drain(ex);
                send(ex, 200, inspect);
            });
            server.createContext("/ml/train", ex -> {
                lastTrainRequest = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                String body = trainTemplate;
                for (String id : templateModelIds) {
                    body = body.replace(id, UUID.randomUUID().toString().replace("-", ""));
                }
                send(ex, 200, body);
            });
            server.createContext("/ml/predict", ex -> {
                drain(ex);
                send(ex, 200, predict);
            });
            server.start();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    String baseUrl() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    void stop() {
        server.stop(0);
    }

    private static String resource(String path) throws IOException {
        try (InputStream in = StubMlServer.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IOException("Missing test resource " + path);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void drain(HttpExchange ex) throws IOException {
        ex.getRequestBody().readAllBytes();
    }

    private static void send(HttpExchange ex, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }
}
