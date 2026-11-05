package edu.ait.diabetesrisk.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/** One run of the training wizard. Never exposed directly; see TrainingJobDto. */
@Entity
@Table(name = "training_job")
public class TrainingJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 200)
    private String originalFilename;

    /** Generated name inside the uploads folder. Internal only. */
    @Column(nullable = false, length = 80)
    private String storedFilename;

    private long fileSizeBytes;

    @Column(nullable = false, length = 20)
    private String datasetOrigin;

    @Column(nullable = false, length = 200)
    private String targetColumn;

    @Convert(converter = StringListConverter.class)
    @Column(length = 20000)
    private List<String> featureColumns = new ArrayList<>();

    private double trainFraction;

    @Convert(converter = StringListConverter.class)
    @Column(length = 200)
    private List<String> algorithms = new ArrayList<>();

    @Column(nullable = false, length = 20)
    private String scoringMetric;

    @Column(nullable = false, length = 20)
    private String tuningLevel;

    private int randomSeed;

    @Column(length = 200)
    private String positiveClass;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private JobStatus status = JobStatus.QUEUED;

    @Column(length = 120)
    private String stage;

    /** Safe, user-facing message only (no paths or stack traces). */
    @Column(length = 1000)
    private String errorMessage;

    @Column(nullable = false)
    private Instant createdAt;

    private Instant startedAt;
    private Instant finishedAt;

    private Integer totalRows;
    private Integer rowsUsed;
    private Integer trainRows;
    private Integer testRows;
    private Long selectedModelId;
    private Double bestCvScore;

    /** Full ML results document (validated JSON returned by the ML service). */
    @Lob
    private String resultsJson;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getOriginalFilename() { return originalFilename; }
    public void setOriginalFilename(String originalFilename) { this.originalFilename = originalFilename; }
    public String getStoredFilename() { return storedFilename; }
    public void setStoredFilename(String storedFilename) { this.storedFilename = storedFilename; }
    public long getFileSizeBytes() { return fileSizeBytes; }
    public void setFileSizeBytes(long fileSizeBytes) { this.fileSizeBytes = fileSizeBytes; }
    public String getDatasetOrigin() { return datasetOrigin; }
    public void setDatasetOrigin(String datasetOrigin) { this.datasetOrigin = datasetOrigin; }
    public String getTargetColumn() { return targetColumn; }
    public void setTargetColumn(String targetColumn) { this.targetColumn = targetColumn; }
    public List<String> getFeatureColumns() { return featureColumns; }
    public void setFeatureColumns(List<String> featureColumns) { this.featureColumns = featureColumns; }
    public double getTrainFraction() { return trainFraction; }
    public void setTrainFraction(double trainFraction) { this.trainFraction = trainFraction; }
    public List<String> getAlgorithms() { return algorithms; }
    public void setAlgorithms(List<String> algorithms) { this.algorithms = algorithms; }
    public String getScoringMetric() { return scoringMetric; }
    public void setScoringMetric(String scoringMetric) { this.scoringMetric = scoringMetric; }
    public String getTuningLevel() { return tuningLevel; }
    public void setTuningLevel(String tuningLevel) { this.tuningLevel = tuningLevel; }
    public int getRandomSeed() { return randomSeed; }
    public void setRandomSeed(int randomSeed) { this.randomSeed = randomSeed; }
    public String getPositiveClass() { return positiveClass; }
    public void setPositiveClass(String positiveClass) { this.positiveClass = positiveClass; }
    public JobStatus getStatus() { return status; }
    public void setStatus(JobStatus status) { this.status = status; }
    public String getStage() { return stage; }
    public void setStage(String stage) { this.stage = stage; }
    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }
    public Instant getFinishedAt() { return finishedAt; }
    public void setFinishedAt(Instant finishedAt) { this.finishedAt = finishedAt; }
    public Integer getTotalRows() { return totalRows; }
    public void setTotalRows(Integer totalRows) { this.totalRows = totalRows; }
    public Integer getRowsUsed() { return rowsUsed; }
    public void setRowsUsed(Integer rowsUsed) { this.rowsUsed = rowsUsed; }
    public Integer getTrainRows() { return trainRows; }
    public void setTrainRows(Integer trainRows) { this.trainRows = trainRows; }
    public Integer getTestRows() { return testRows; }
    public void setTestRows(Integer testRows) { this.testRows = testRows; }
    public Long getSelectedModelId() { return selectedModelId; }
    public void setSelectedModelId(Long selectedModelId) { this.selectedModelId = selectedModelId; }
    public Double getBestCvScore() { return bestCvScore; }
    public void setBestCvScore(Double bestCvScore) { this.bestCvScore = bestCvScore; }
    public String getResultsJson() { return resultsJson; }
    public void setResultsJson(String resultsJson) { this.resultsJson = resultsJson; }
}
