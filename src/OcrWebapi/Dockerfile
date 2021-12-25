#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM emgu/bazel-android:dotnet-6.0-bazel-4.2.1 AS base
RUN apt-get update  
RUN apt-get install -y apt-utils  
RUN apt-get install -y libgdiplus  

WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["OcrWebapi/OcrWebapi.csproj", "OcrWebapi/"]
RUN dotnet restore "OcrWebapi/OcrWebapi.csproj"
COPY . .
WORKDIR "/src/OcrWebapi"
RUN dotnet build "OcrWebapi.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "OcrWebapi.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "OcrWebapi.dll"]