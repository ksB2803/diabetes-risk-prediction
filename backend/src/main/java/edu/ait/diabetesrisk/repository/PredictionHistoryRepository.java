package edu.ait.diabetesrisk.repository;

import edu.ait.diabetesrisk.entity.PredictionHistory;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PredictionHistoryRepository extends JpaRepository<PredictionHistory, Long> {

    List<PredictionHistory> findAllByOrderByCreatedAtDesc(Pageable pageable);
}
