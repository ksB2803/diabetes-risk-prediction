package edu.ait.diabetesrisk.config;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

/** Typed view of the {@code app.*} properties in application.properties. */
@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private final Storage storage = new Storage();
    private final Ml ml = new Ml();
    private final Training training = new Training();
    private final Cors cors = new Cors();

    public Storage getStorage() { return storage; }
    public Ml getMl() { return ml; }
    public Training getTraining() { return training; }
    public Cors getCors() { return cors; }

    public static class Storage {
        private String dir = "./storage";
        private boolean keepUploads = true;

        public String getDir() { return dir; }
        public void setDir(String dir) { this.dir = dir; }
        public boolean isKeepUploads() { return keepUploads; }
        public void setKeepUploads(boolean keepUploads) { this.keepUploads = keepUploads; }
    }

    public static class Ml {
        private String baseUrl = "http://127.0.0.1:8000";
        private String apiKey = "";
        private Duration connectTimeout = Duration.ofSeconds(3);
        private Duration requestTimeout = Duration.ofSeconds(30);
        private Duration trainingTimeout = Duration.ofMinutes(10);

        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getApiKey() { return apiKey; }
        public void setApiKey(String apiKey) { this.apiKey = apiKey; }
        public Duration getConnectTimeout() { return connectTimeout; }
        public void setConnectTimeout(Duration connectTimeout) { this.connectTimeout = connectTimeout; }
        public Duration getRequestTimeout() { return requestTimeout; }
        public void setRequestTimeout(Duration requestTimeout) { this.requestTimeout = requestTimeout; }
        public Duration getTrainingTimeout() { return trainingTimeout; }
        public void setTrainingTimeout(Duration trainingTimeout) { this.trainingTimeout = trainingTimeout; }
    }

    public static class Training {
        private int maxConcurrentJobs = 1;
        private int queueCapacity = 10;

        public int getMaxConcurrentJobs() { return maxConcurrentJobs; }
        public void setMaxConcurrentJobs(int maxConcurrentJobs) { this.maxConcurrentJobs = maxConcurrentJobs; }
        public int getQueueCapacity() { return queueCapacity; }
        public void setQueueCapacity(int queueCapacity) { this.queueCapacity = queueCapacity; }
    }

    public static class Cors {
        private List<String> allowedOrigins = new ArrayList<>();

        public List<String> getAllowedOrigins() { return allowedOrigins; }
        public void setAllowedOrigins(List<String> allowedOrigins) { this.allowedOrigins = allowedOrigins; }
    }
}
