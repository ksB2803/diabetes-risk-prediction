package edu.ait.diabetesrisk.service;

import edu.ait.diabetesrisk.config.AppProperties;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.UUID;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

/**
 * Controlled local storage for uploaded CSVs. File names are generated here
 * (random UUIDs); user-supplied names are never used as paths.
 */
@Service
public class StorageService {

    private static final Pattern STORED_NAME = Pattern.compile("^[a-f0-9-]{36}\\.csv$");

    private final Path uploadsDir;

    public StorageService(AppProperties props) {
        this.uploadsDir = Path.of(props.getStorage().getDir()).toAbsolutePath().normalize().resolve("uploads");
        try {
            Files.createDirectories(uploadsDir);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not create the uploads directory", e);
        }
    }

    /** Writes the bytes atomically and returns the generated file name. */
    public String storeUpload(byte[] content) {
        String name = UUID.randomUUID() + ".csv";
        Path target = resolve(name);
        try {
            Path tmp = Files.createTempFile(uploadsDir, ".upload-", ".tmp");
            Files.write(tmp, content);
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not store the uploaded file", e);
        }
        return name;
    }

    /** Resolves a generated name, refusing anything that could escape the uploads folder. */
    public Path resolve(String storedName) {
        if (storedName == null || !STORED_NAME.matcher(storedName).matches()) {
            throw new IllegalArgumentException("Invalid stored file name");
        }
        Path path = uploadsDir.resolve(storedName).normalize();
        if (!path.startsWith(uploadsDir)) {
            throw new IllegalArgumentException("Invalid stored file name");
        }
        return path;
    }

    public void deleteQuietly(String storedName) {
        try {
            Files.deleteIfExists(resolve(storedName));
        } catch (IOException | IllegalArgumentException ignored) {
            // Best effort cleanup.
        }
    }
}
