package edu.ait.diabetesrisk;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

/** A real HTTP upload above the configured limit returns 413 with the JSON error body. */
@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {
                "spring.servlet.multipart.max-file-size=1KB",
                "spring.servlet.multipart.max-request-size=2KB",
                "spring.datasource.url=jdbc:h2:mem:upload-limit-test;DB_CLOSE_DELAY=-1",
                "app.ml.base-url=http://127.0.0.1:9"
        })
class UploadLimitTest {

    @Autowired
    TestRestTemplate rest;

    @Test
    void oversizedUploadReturns413() {
        byte[] big = "a,b\n1,2\n".repeat(1000).getBytes(StandardCharsets.UTF_8);
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", new ByteArrayResource(big) {
            @Override
            public String getFilename() {
                return "big.csv";
            }
        });
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        ResponseEntity<String> response = rest.postForEntity("/api/datasets/inspect", new HttpEntity<>(parts, headers), String.class);
        assertThat(response.getStatusCode().value()).isEqualTo(413);
        assertThat(response.getBody()).contains("FILE_TOO_LARGE");
    }
}
