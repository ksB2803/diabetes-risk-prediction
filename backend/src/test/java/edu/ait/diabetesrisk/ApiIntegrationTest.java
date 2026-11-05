package edu.ait.diabetesrisk;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest
@AutoConfigureMockMvc
class ApiIntegrationTest {

    private static StubMlServer stub;
    private static Path storageDir;

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) throws Exception {
        stub = new StubMlServer();
        storageDir = Files.createTempDirectory("drp-test-storage");
        registry.add("app.ml.base-url", stub::baseUrl);
        registry.add("app.storage.dir", storageDir::toString);
        registry.add("spring.datasource.url", () -> "jdbc:h2:mem:api-test;DB_CLOSE_DELAY=-1");
    }

    @AfterAll
    static void stopStub() {
        stub.stop();
    }

    @Autowired
    MockMvc mvc;

    @Autowired
    ObjectMapper mapper;

    @Autowired
    JdbcTemplate jdbc;

    private static final byte[] CSV = ("Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction,Age,"
            + "PhysicalActivity,FamilyHistory,Diabetes\n2,150,70,33.1,0.5,45,Low,Yes,1\n1,90,66,24.0,0.2,30,High,No,0\n")
            .getBytes(StandardCharsets.UTF_8);

    private static final List<String> FEATURES = List.of("Pregnancies", "Glucose", "BloodPressure", "BMI",
            "DiabetesPedigreeFunction", "Age", "PhysicalActivity", "FamilyHistory");

    private MockMultipartFile csv(String name, byte[] bytes) {
        return new MockMultipartFile("file", name, "text/csv", bytes);
    }

    private MockMultipartFile config(Map<String, Object> overrides) throws Exception {
        Map<String, Object> cfg = new java.util.LinkedHashMap<>();
        cfg.put("targetColumn", "Diabetes");
        cfg.put("featureColumns", FEATURES);
        cfg.put("trainFraction", 0.8);
        cfg.put("algorithms", List.of("logistic_regression", "random_forest"));
        cfg.put("scoringMetric", "roc_auc");
        cfg.put("tuningLevel", "balanced");
        cfg.put("randomSeed", 42);
        cfg.put("datasetOrigin", "synthetic");
        cfg.putAll(overrides);
        return new MockMultipartFile("config", "", MediaType.APPLICATION_JSON_VALUE, mapper.writeValueAsBytes(cfg));
    }

    private JsonNode json(MvcResult result) throws Exception {
        return mapper.readTree(result.getResponse().getContentAsString());
    }

    private void assertNoLeaks(String body) {
        assertThat(body).doesNotContain(storageDir.toString());
        assertThat(body).doesNotContain(".joblib");
        assertThat(body).doesNotContain("Exception");
        assertThat(body).doesNotContain("at edu.ait");
    }

    /** Creates a job, waits for the worker, returns the completed job JSON. */
    private JsonNode trainAndWait() throws Exception {
        MvcResult created = mvc.perform(multipart("/api/training-jobs").file(csv("demo.csv", CSV)).file(config(Map.of())))
                .andExpect(status().isAccepted())
                .andExpect(header().exists("Location"))
                .andReturn();
        long id = json(created).path("id").asLong();
        JsonNode job = null;
        for (int i = 0; i < 100; i++) {
            job = json(mvc.perform(get("/api/training-jobs/" + id)).andExpect(status().isOk()).andReturn());
            String s = job.path("status").asText();
            if (s.equals("COMPLETED") || s.equals("FAILED")) {
                break;
            }
            Thread.sleep(100);
        }
        assertThat(job.path("status").asText()).isEqualTo("COMPLETED");
        return job;
    }

    @Test
    void schemaTablesAreCreatedInH2() {
        List<String> tables = jdbc.queryForList(
                "SELECT LOWER(TABLE_NAME) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'PUBLIC'", String.class);
        assertThat(tables).contains("training_job", "trained_model", "prediction_history");
    }

    @Test
    void healthReportsDatabaseAndMlService() throws Exception {
        mvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"))
                .andExpect(jsonPath("$.database").value("UP"))
                .andExpect(jsonPath("$.mlService").value("UP"));
    }

    @Test
    void inspectReturnsProfile() throws Exception {
        mvc.perform(multipart("/api/datasets/inspect").file(csv("demo.csv", CSV)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.suggested_target").value("Diabetes"))
                .andExpect(jsonPath("$.columns").isArray());
    }

    @Test
    void inspectRejectsEmptyFile() throws Exception {
        mvc.perform(multipart("/api/datasets/inspect").file(csv("empty.csv", new byte[0])))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("EMPTY_FILE"));
    }

    @Test
    void inspectRejectsNonCsvName() throws Exception {
        mvc.perform(multipart("/api/datasets/inspect").file(csv("data.xlsx", CSV)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("NOT_A_CSV"));
    }

    @Test
    void inspectRejectsBinaryContentWithCsvName() throws Exception {
        byte[] binary = {(byte) 0x89, 'P', 'N', 'G', 0, 0, 0, 1};
        mvc.perform(multipart("/api/datasets/inspect").file(csv("fake.csv", binary)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("NOT_A_CSV"));
    }

    @Test
    void fullFlowTrainResultsPredictHistory() throws Exception {
        JsonNode job = trainAndWait();
        long jobId = job.path("id").asLong();
        assertThat(job.path("trainRows").asInt()).isPositive();
        assertThat(job.path("selectedModelId").isNumber()).isTrue();
        assertThat(job.has("storedFilename")).isFalse();
        assertThat(stub.lastTrainRequest).contains("\"target_column\":\"Diabetes\"").contains("\"tuning_level\":\"balanced\"");

        MvcResult resultsResponse = mvc.perform(get("/api/training-jobs/" + jobId + "/results"))
                .andExpect(status().isOk())
                .andReturn();
        String resultsBody = resultsResponse.getResponse().getContentAsString();
        assertNoLeaks(resultsBody);
        JsonNode results = mapper.readTree(resultsBody);
        assertThat(results.path("models").size()).isEqualTo(2);
        JsonNode first = results.path("models").get(0);
        assertThat(first.has("model_id")).isFalse();
        assertThat(first.path("artifact").has("file")).isFalse();
        assertThat(first.path("id").isNumber()).isTrue();
        assertThat(first.path("cv_score").isNumber()).isTrue();
        assertThat(first.path("test_metrics").path("roc_auc").isNumber()).isTrue();
        assertThat(results.path("job").path("id").asLong()).isEqualTo(jobId);

        mvc.perform(get("/api/training-jobs/" + jobId + "/report").param("format", "csv"))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith("text/csv"))
                .andExpect(content().string(org.hamcrest.Matchers.containsString("cv_score")));

        long modelId = first.path("id").asLong();
        MvcResult modelResponse = mvc.perform(get("/api/models/" + modelId)).andExpect(status().isOk()).andReturn();
        assertNoLeaks(modelResponse.getResponse().getContentAsString());
        JsonNode model = json(modelResponse);
        assertThat(model.path("features").size()).isEqualTo(FEATURES.size());
        assertThat(model.has("mlModelId")).isFalse();

        Map<String, Object> values = new java.util.LinkedHashMap<>();
        values.put("Pregnancies", 2);
        values.put("Glucose", 150);
        values.put("BloodPressure", 70);
        values.put("BMI", 33.1);
        values.put("DiabetesPedigreeFunction", 0.5);
        values.put("Age", 45);
        values.put("PhysicalActivity", "Low");
        values.put("FamilyHistory", "Yes");
        mvc.perform(post("/api/models/" + modelId + "/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(mapper.writeValueAsBytes(Map.of("values", values))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.predictedClass").exists())
                .andExpect(jsonPath("$.disclaimer").value(org.hamcrest.Matchers.containsString("not a medical device")));

        mvc.perform(get("/api/predictions"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].modelId").value(modelId));
        // Inputs are not stored: the history table has no column for them.
        List<String> columns = jdbc.queryForList(
                "SELECT LOWER(COLUMN_NAME) FROM INFORMATION_SCHEMA.COLUMNS WHERE LOWER(TABLE_NAME) = 'prediction_history'",
                String.class);
        assertThat(columns).noneMatch(c -> c.contains("glucose") || c.contains("input") || c.contains("values"));

        // Incompatible schema is rejected with a clear message.
        mvc.perform(post("/api/models/" + modelId + "/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"values\":{\"Glucose\":120,\"Insulin\":80}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INCOMPATIBLE_FEATURES"));

        // Wrong type is rejected.
        values.put("Glucose", "high");
        mvc.perform(post("/api/models/" + modelId + "/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(mapper.writeValueAsBytes(Map.of("values", values))))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_VALUE"));

        mvc.perform(get("/api/summary"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.completedJobs").value(org.hamcrest.Matchers.greaterThanOrEqualTo(1)));
    }

    @Test
    void invalidTrainingConfigIsRejected() throws Exception {
        mvc.perform(multipart("/api/training-jobs").file(csv("demo.csv", CSV)).file(config(Map.of("trainFraction", 0.95))))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("VALIDATION_ERROR"));

        mvc.perform(multipart("/api/training-jobs").file(csv("demo.csv", CSV))
                        .file(config(Map.of("algorithms", List.of("svm")))))
                .andExpect(status().isBadRequest());

        mvc.perform(multipart("/api/training-jobs").file(csv("demo.csv", CSV))
                        .file(config(Map.of("featureColumns", List.of("Glucose", "Diabetes")))))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_CONFIG"));

        mvc.perform(multipart("/api/training-jobs").file(csv("demo.csv", CSV)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("MISSING_PART"));
    }

    @Test
    void missingResourcesReturn404() throws Exception {
        mvc.perform(get("/api/training-jobs/987654")).andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("NOT_FOUND"));
        mvc.perform(get("/api/training-jobs/987654/results")).andExpect(status().isNotFound());
        mvc.perform(get("/api/models/987654")).andExpect(status().isNotFound());
        mvc.perform(post("/api/models/987654/predict").contentType(MediaType.APPLICATION_JSON)
                .content("{\"values\":{\"a\":1}}")).andExpect(status().isNotFound());
    }

    @Test
    void malformedJsonIsA400WithoutStackTrace() throws Exception {
        MvcResult r = mvc.perform(post("/api/models/1/predict").contentType(MediaType.APPLICATION_JSON).content("{not json"))
                .andExpect(status().isBadRequest())
                .andReturn();
        assertNoLeaks(r.getResponse().getContentAsString());
    }

    @Test
    void staticFrontendIsServed() throws Exception {
        for (String page : List.of("/index.html", "/train.html", "/results.html", "/predict.html", "/history.html")) {
            mvc.perform(get(page))
                    .andExpect(status().isOk())
                    .andExpect(content().string(org.hamcrest.Matchers.containsString("AI/ML Mini Project")));
        }
    }
}
