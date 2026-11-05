package edu.ait.diabetesrisk.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import java.time.Instant;

/** A fitted preprocessing + classifier pipeline saved by the ML service. */
@Entity
@Table(name = "trained_model", indexes = @Index(name = "idx_model_job", columnList = "trainingJobId"))
public class TrainedModel {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long trainingJobId;

    /** Artifact ID inside the ML service. Internal only. */
    @Column(nullable = false, unique = true, length = 40)
    private String mlModelId;

    /** Artifact file name inside the ML model store. Internal only. */
    @Column(nullable = false, length = 80)
    private String artifactFile;

    @Column(nullable = false, length = 40)
    private String algorithm;

    @Column(nullable = false, length = 60)
    private String displayName;

    @Column(length = 60)
    private String version;

    @Column(nullable = false, length = 200)
    private String targetColumn;

    @Column(length = 200)
    private String positiveClass;

    @Column(length = 200)
    private String negativeClass;

    /** JSON list of {name, type, min, max, median, categories}. */
    @Lob
    private String featureSchemaJson;

    @Column(nullable = false, length = 20)
    private String selectionMetric;

    private Double cvScore;
    private Double cvStd;

    /** JSON object of held-out test metrics. */
    @Lob
    private String testMetricsJson;

    @Lob
    private String bestParamsJson;

    private boolean selectedByCv;
    private boolean available = true;

    @Column(nullable = false)
    private Instant createdAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getTrainingJobId() { return trainingJobId; }
    public void setTrainingJobId(Long trainingJobId) { this.trainingJobId = trainingJobId; }
    public String getMlModelId() { return mlModelId; }
    public void setMlModelId(String mlModelId) { this.mlModelId = mlModelId; }
    public String getArtifactFile() { return artifactFile; }
    public void setArtifactFile(String artifactFile) { this.artifactFile = artifactFile; }
    public String getAlgorithm() { return algorithm; }
    public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }
    public String getDisplayName() { return displayName; }
    public void setDisplayName(String displayName) { this.displayName = displayName; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public String getTargetColumn() { return targetColumn; }
    public void setTargetColumn(String targetColumn) { this.targetColumn = targetColumn; }
    public String getPositiveClass() { return positiveClass; }
    public void setPositiveClass(String positiveClass) { this.positiveClass = positiveClass; }
    public String getNegativeClass() { return negativeClass; }
    public void setNegativeClass(String negativeClass) { this.negativeClass = negativeClass; }
    public String getFeatureSchemaJson() { return featureSchemaJson; }
    public void setFeatureSchemaJson(String featureSchemaJson) { this.featureSchemaJson = featureSchemaJson; }
    public String getSelectionMetric() { return selectionMetric; }
    public void setSelectionMetric(String selectionMetric) { this.selectionMetric = selectionMetric; }
    public Double getCvScore() { return cvScore; }
    public void setCvScore(Double cvScore) { this.cvScore = cvScore; }
    public Double getCvStd() { return cvStd; }
    public void setCvStd(Double cvStd) { this.cvStd = cvStd; }
    public String getTestMetricsJson() { return testMetricsJson; }
    public void setTestMetricsJson(String testMetricsJson) { this.testMetricsJson = testMetricsJson; }
    public String getBestParamsJson() { return bestParamsJson; }
    public void setBestParamsJson(String bestParamsJson) { this.bestParamsJson = bestParamsJson; }
    public boolean isSelectedByCv() { return selectedByCv; }
    public void setSelectedByCv(boolean selectedByCv) { this.selectedByCv = selectedByCv; }
    public boolean isAvailable() { return available; }
    public void setAvailable(boolean available) { this.available = available; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
