package edu.ait.diabetesrisk.repository;

import edu.ait.diabetesrisk.entity.TrainedModel;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TrainedModelRepository extends JpaRepository<TrainedModel, Long> {

    List<TrainedModel> findByTrainingJobIdOrderByIdAsc(Long trainingJobId);

    List<TrainedModel> findAllByOrderByCreatedAtDesc();

    long countByTrainingJobId(Long trainingJobId);

    long countByAvailableTrue();
}
