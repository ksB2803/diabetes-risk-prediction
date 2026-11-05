package edu.ait.diabetesrisk.controller;

import com.fasterxml.jackson.databind.JsonNode;
import edu.ait.diabetesrisk.dto.TrainingConfigRequest;
import edu.ait.diabetesrisk.dto.TrainingJobDto;
import edu.ait.diabetesrisk.exception.ApiException;
import edu.ait.diabetesrisk.service.TrainingJobService;
import jakarta.validation.Valid;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class TrainingJobController {

    private final TrainingJobService service;

    public TrainingJobController(TrainingJobService service) {
        this.service = service;
    }

    /** Wizard step 1: profile a CSV (rows, columns, preview, types, missing values). Nothing is stored. */
    @PostMapping(path = "/datasets/inspect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public JsonNode inspect(@RequestPart("file") MultipartFile file) {
        return service.inspect(file);
    }

    /** Start a training job: multipart with `file` (CSV) and `config` (JSON). Returns 202 + job. */
    @PostMapping(path = "/training-jobs", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<TrainingJobDto> create(@RequestPart("file") MultipartFile file,
                                                 @Valid @RequestPart("config") TrainingConfigRequest config) {
        TrainingJobDto job = service.create(file, config);
        return ResponseEntity.accepted().location(URI.create("/api/training-jobs/" + job.id())).body(job);
    }

    @GetMapping("/training-jobs")
    public List<TrainingJobDto> list(@RequestParam(defaultValue = "50") int limit) {
        return service.list(limit);
    }

    @GetMapping("/training-jobs/{id}")
    public TrainingJobDto get(@PathVariable Long id) {
        return service.get(id);
    }

    @GetMapping("/training-jobs/{id}/results")
    public JsonNode results(@PathVariable Long id) {
        return service.results(id);
    }

    /** Downloadable report: ?format=json (default) or ?format=csv. */
    @GetMapping("/training-jobs/{id}/report")
    public ResponseEntity<?> report(@PathVariable Long id, @RequestParam(defaultValue = "json") String format) {
        if ("csv".equalsIgnoreCase(format)) {
            String csv = service.csvReport(id);
            return ResponseEntity.ok()
                    .contentType(new MediaType("text", "csv", StandardCharsets.UTF_8))
                    .header(HttpHeaders.CONTENT_DISPOSITION, attachment("training-job-" + id + "-report.csv"))
                    .body(csv);
        }
        if (!"json".equalsIgnoreCase(format)) {
            throw ApiException.badRequest("BAD_FORMAT", "Report format must be json or csv.");
        }
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .header(HttpHeaders.CONTENT_DISPOSITION, attachment("training-job-" + id + "-report.json"))
                .body(service.results(id));
    }

    private static String attachment(String filename) {
        return ContentDisposition.attachment().filename(filename).build().toString();
    }
}
