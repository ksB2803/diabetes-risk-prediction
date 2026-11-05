package edu.ait.diabetesrisk.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/** CORS (opt-in) and basic security headers for every response. */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    private final AppProperties props;

    public WebConfig(AppProperties props) {
        this.props = props;
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        var origins = props.getCors().getAllowedOrigins().stream().filter(o -> o != null && !o.isBlank()).toList();
        if (!origins.isEmpty()) {
            registry.addMapping("/api/**")
                    .allowedOrigins(origins.toArray(String[]::new))
                    .allowedMethods("GET", "POST")
                    .maxAge(3600);
        }
    }

    @Bean
    public OncePerRequestFilter securityHeadersFilter() {
        return new OncePerRequestFilter() {
            @Override
            protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
                    throws ServletException, IOException {
                response.setHeader("X-Content-Type-Options", "nosniff");
                response.setHeader("Referrer-Policy", "no-referrer");
                if (!request.getRequestURI().startsWith("/h2-console")) {
                    response.setHeader("X-Frame-Options", "DENY");
                    response.setHeader("Content-Security-Policy",
                            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                    + "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
                                    + "frame-ancestors 'none'; base-uri 'self'; form-action 'self'");
                }
                chain.doFilter(request, response);
            }
        };
    }
}
