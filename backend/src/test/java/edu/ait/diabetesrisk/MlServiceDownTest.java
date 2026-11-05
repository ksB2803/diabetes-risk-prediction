package edu.ait.diabetesrisk;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/** The ML service is down: every ML-backed call degrades to a clear 503 or a failed job. */
@SpringBootTest
@AutoConfigureMockMvc
class MlServiceDownTest {

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) throws IOException {
        int closedPort;
        try (ServerSocket socket = new ServerSocket(0)) {
            closedPort = socket.getLocalPort();
        }
        registry.add("app.ml.base-url", () -> "http://127.0.0.1:" + closedPort);
        registry.add("app.storage.dir", () -> System.getProperty("java.io.tmpdir") + "/drp-down-test");
        registry.add("spring.datasource.url", () -> "jdbc:h2:mem:down-test;DB_CLOSE_DELAY=-1");
    }

    @Autowired
    MockMvc mvc;

    @Autowired
    ObjectMapper mapper;

    private static final byte[] CSV = "a,b,Diabetes\n1,2,0\n3,4,1\n".getBytes(StandardCharsets.UTF_8);

    @Test
    void healthShowsMlDown() throws Exception {
        mvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("DEGRADED"))
                .andExpect(jsonPath("$.mlService").value("DOWN"));
    }

    @Test
    void inspectReturns503() throws Exception {
        MvcResult r = mvc.perform(multipart("/api/datasets/inspect")
                        .file(new MockMultipartFile("file", "d.csv", "text/csv", CSV)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("ML_SERVICE_UNAVAILABLE"))
                .andReturn();
        assertThat(r.getResponse().getContentAsString()).doesNotContain("Exception").doesNotContain("127.0.0.1");
    }

    @Test
    void trainingJobFailsWithSafeMessage() throws Exception {
        Map<String, Object> cfg = Map.of(
                "targetColumn", "Diabetes",
                "featureColumns", List.of("a", "b"),
                "trainFraction", 0.8,
                "algorithms", List.of("logistic_regression"),
                "scoringMetric", "roc_auc",
                "tuningLevel", "quick",
                "randomSeed", 42);
        MvcResult created = mvc.perform(multipart("/api/training-jobs")
                        .file(new MockMultipartFile("file", "d.csv", "text/csv", CSV))
                        .file(new MockMultipartFile("config", "", MediaType.APPLICATION_JSON_VALUE, mapper.writeValueAsBytes(cfg))))
                .andExpect(status().isAccepted())
                .andReturn();
        long id = mapper.readTree(created.getResponse().getContentAsString()).path("id").asLong();

        JsonNode job = null;
        for (int i = 0; i < 100; i++) {
            job = mapper.readTree(mvc.perform(get("/api/training-jobs/" + id)).andReturn().getResponse().getContentAsString());
            if ("FAILED".equals(job.path("status").asText())) {
                break;
            }
            Thread.sleep(100);
        }
        assertThat(job.path("status").asText()).isEqualTo("FAILED");
        assertThat(job.path("errorMessage").asText()).contains("not reachable").doesNotContain("Exception");

        mvc.perform(get("/api/training-jobs/" + id + "/results"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("RESULTS_NOT_READY"));
    }
}
