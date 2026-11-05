package edu.ait.diabetesrisk.controller;

import edu.ait.diabetesrisk.dto.PredictionHistoryDto;
import edu.ait.diabetesrisk.dto.PredictionRequest;
import edu.ait.diabetesrisk.dto.PredictionResponseDto;
import edu.ait.diabetesrisk.dto.TrainedModelDto;
import edu.ait.diabetesrisk.service.ModelService;
import edu.ait.diabetesrisk.service.PredictionService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class ModelController {

    private final ModelService models;
    private final PredictionService predictions;

    public ModelController(ModelService models, PredictionService predictions) {
        this.models = models;
        this.predictions = predictions;
    }

    @GetMapping("/models")
    public List<TrainedModelDto> list(@RequestParam(defaultValue = "false") boolean includeUnavailable) {
        return models.list(includeUnavailable);
    }

    @GetMapping("/models/{id}")
    public TrainedModelDto get(@PathVariable Long id) {
        return models.get(id);
    }

    @PostMapping("/models/{id}/predict")
    public PredictionResponseDto predict(@PathVariable Long id, @Valid @RequestBody PredictionRequest request) {
        return predictions.predict(id, request);
    }

    /** Recent predictions (class and probability only; inputs are never stored). */
    @GetMapping("/predictions")
    public List<PredictionHistoryDto> recentPredictions(@RequestParam(defaultValue = "20") int limit) {
        return predictions.recent(limit);
    }
}
