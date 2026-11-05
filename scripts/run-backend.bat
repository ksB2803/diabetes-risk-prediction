@echo off
REM Start the Spring Boot backend + frontend on http://localhost:8080 (Windows).
REM Uses the "local" profile, which enables the H2 console for localhost only.
cd /d "%~dp0..\backend"
mvn spring-boot:run -Dspring-boot.run.profiles=local
