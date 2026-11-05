package edu.ait.diabetesrisk.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * A record that a prediction was made. Input values are deliberately NOT stored,
 * because they may be sensitive health information.
 */
@Entity
@Table(name = "prediction_history")
public class PredictionHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long modelId;

    @Column(nullable = false, length = 40)
    private String algorithm;

    @Column(nullable = false, length = 200)
    private String predictedClass;

    private boolean positiveClass;

    private Double probabilityPositive;

    @Column(nullable = false)
    private Instant createdAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getModelId() { return modelId; }
    public void setModelId(Long modelId) { this.modelId = modelId; }
    public String getAlgorithm() { return algorithm; }
    public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }
    public String getPredictedClass() { return predictedClass; }
    public void setPredictedClass(String predictedClass) { this.predictedClass = predictedClass; }
    public boolean isPositiveClass() { return positiveClass; }
    public void setPositiveClass(boolean positiveClass) { this.positiveClass = positiveClass; }
    public Double getProbabilityPositive() { return probabilityPositive; }
    public void setProbabilityPositive(Double probabilityPositive) { this.probabilityPositive = probabilityPositive; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
