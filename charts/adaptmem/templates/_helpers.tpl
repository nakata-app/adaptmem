{{/*
Expand the chart name.
*/}}
{{- define "adaptmem.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name. release-name + chart-name, k8s-safe.
*/}}
{{- define "adaptmem.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "adaptmem.labels" -}}
app.kubernetes.io/name: {{ include "adaptmem.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app: {{ include "adaptmem.name" . }}
{{- end -}}

{{/*
Selector labels — must remain stable across upgrades, so do NOT
include version / chart fields.
*/}}
{{- define "adaptmem.selectorLabels" -}}
app.kubernetes.io/name: {{ include "adaptmem.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: {{ include "adaptmem.name" . }}
{{- end -}}

{{/*
Image reference — uses Chart.appVersion when image.tag is empty.
*/}}
{{- define "adaptmem.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{ printf "%s:%s" .Values.image.repository $tag }}
{{- end -}}

{{/*
Secret name — existingSecret takes precedence; otherwise the chart-managed name.
*/}}
{{- define "adaptmem.secretName" -}}
{{- if .Values.auth.existingSecret -}}
{{ .Values.auth.existingSecret }}
{{- else -}}
{{ printf "%s-secret" (include "adaptmem.fullname" .) }}
{{- end -}}
{{- end -}}
