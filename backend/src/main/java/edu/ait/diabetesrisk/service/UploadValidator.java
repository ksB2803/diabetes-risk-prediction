package edu.ait.diabetesrisk.service;

import edu.ait.diabetesrisk.exception.ApiException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

/**
 * First-line checks on an uploaded CSV. The file name and content type are not
 * trusted; the bytes are checked to look like text. Full parsing and schema
 * validation happen in the ML service.
 */
@Component
public class UploadValidator {

    private static final int SNIFF_BYTES = 64 * 1024;

    public byte[] readCsv(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw ApiException.badRequest("EMPTY_FILE", "Choose a non-empty CSV file to upload.");
        }
        String name = displayName(file.getOriginalFilename());
        if (!name.toLowerCase(Locale.ROOT).endsWith(".csv")) {
            throw ApiException.badRequest("NOT_A_CSV", "Only .csv files are accepted.");
        }
        byte[] bytes;
        try {
            bytes = file.getBytes();
        } catch (IOException e) {
            throw new UncheckedIOException("Could not read upload", e);
        }
        int n = Math.min(bytes.length, SNIFF_BYTES);
        for (int i = 0; i < n; i++) {
            if (bytes[i] == 0) {
                throw ApiException.badRequest("NOT_A_CSV", "The file looks like binary data, not a CSV text file.");
            }
        }
        if (!looksLikeText(bytes, n)) {
            throw ApiException.badRequest("BAD_ENCODING", "The file is not readable text. Save it as a UTF-8 CSV.");
        }
        return bytes;
    }

    /** Lenient UTF-8 check; falls back to accepting single-byte encodings with few control characters. */
    private static boolean looksLikeText(byte[] bytes, int n) {
        try {
            StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes, 0, Math.max(0, n - 4)));
            return true;
        } catch (CharacterCodingException e) {
            int control = 0;
            for (int i = 0; i < n; i++) {
                int b = bytes[i] & 0xff;
                if (b < 0x09 || (b > 0x0d && b < 0x20)) {
                    control++;
                }
            }
            return control < n / 100 + 1;
        }
    }

    /** Safe display name: no directories, no control characters, bounded length. */
    public static String displayName(String original) {
        if (original == null || original.isBlank()) {
            return "upload.csv";
        }
        String base = original.replace('\\', '/');
        base = base.substring(base.lastIndexOf('/') + 1);
        base = base.replaceAll("[\\p{Cntrl}]", "").trim();
        if (base.isEmpty()) {
            base = "upload.csv";
        }
        return base.length() > 120 ? base.substring(base.length() - 120) : base;
    }
}
