package edu.ait.diabetesrisk.repository;

import edu.ait.diabetesrisk.entity.JobStatus;
import edu.ait.diabetesrisk.entity.TrainingJob;
import java.util.Collection;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TrainingJobRepository extends JpaRepository<TrainingJob, Long> {

    List<TrainingJob> findAllByOrderByCreatedAtDesc(Pageable pageable);

    List<TrainingJob> findByStatusIn(Collection<JobStatus> statuses);

    long countByStatus(JobStatus status);
}
