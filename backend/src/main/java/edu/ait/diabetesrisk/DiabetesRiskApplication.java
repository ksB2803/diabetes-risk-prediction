package edu.ait.diabetesrisk;

import edu.ait.diabetesrisk.config.AppProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

/**
 * Diabetes Risk Prediction - AI/ML Mini Project (AIT).
 * Serves the static frontend, exposes the public REST API and delegates ML work
 * to the internal Python service.
 */
@SpringBootApplication
@EnableConfigurationProperties(AppProperties.class)
public class DiabetesRiskApplication {

    public static void main(String[] args) {
        SpringApplication.run(DiabetesRiskApplication.class, args);
    }
}
